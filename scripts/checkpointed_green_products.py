"""Disk-streamed Green products with immutable, per-transition proof records.

The HVP callback must return genuine enclosures, not point estimates. A
complete manifest is written only after every row and residual is checked.
Interrupted writes leave unreferenced temporary files, never completed rows.
This module does not read training outcomes or assert an event certificate.
"""
import hashlib
import io
import json
import math
import os
from pathlib import Path
import tempfile
import time

import numpy as np
from flint import arb, ctx
from outward_green_momentum import adjoint_hvp_direction, rounded_recurrence_step
from outward_green_products import sequence_upper, streaming_norm_upper


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def json_bytes(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False)+"\n").encode()


def atomic_file(path, writer):
    """Publish without overwriting, accepting only an identical existing file."""
    path=Path(path);path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix=path.name+".", suffix=".partial",
                                     dir=path.parent, delete=False) as stream:
        temporary=Path(stream.name)
        writer(stream);stream.flush();os.fsync(stream.fileno())
    try:
        try:
            # Unlike replace(), this fails atomically if destination exists.
            os.link(temporary, path)
        except FileExistsError:
            if digest(temporary)!=digest(path):
                raise ValueError(f"immutable record differs: {path}")
    finally:
        temporary.unlink()  # Only the temporary file created above.


def save_json(path, value):
    data=json_bytes(value);atomic_file(path, lambda stream: stream.write(data))


def save_array(path, value):
    atomic_file(path, lambda stream: np.save(stream, value, allow_pickle=False))


class StoredRows:
    """A completed, hash-authenticated row sequence; no whole-path allocation."""
    def __init__(self, folder):
        self.folder=Path(folder)
        self.summary=json.loads((self.folder/"summary.json").read_text())
        self.identity=digest(self.folder/"summary.json")
        self.method=json.loads((self.folder/"method.json").read_text())
        if self.summary["method_sha256"]!=digest(self.folder/"method.json"):
            raise ValueError("sequence method changed")
        self.H,self.d=map(int,self.summary["shape"])
        if (self.summary["kind"]!=self.method["kind"] or
            self.summary["shape"]!=self.method["shape"] or
            self.summary["kind"] not in ("input_rows","forward","adjoint")):
            raise ValueError("sequence kind or shape differs from sealed method")
        if self.H<1 or self.d<2 or self.d%2 or len(self.summary["rows"])!=self.H:
            raise ValueError("invalid complete row manifest")
        if self.summary.get("complete") is not True:
            raise ValueError("incomplete input sequence")
        for j,row in enumerate(self.summary["rows"]):
            if row["index"]!=j or row["file"]!=f"row_{j:03d}.npy":
                raise ValueError("noncanonical row manifest")
            if digest(self.folder/row["file"])!=row["sha256"]:
                raise ValueError("input row changed")
        if self.summary["kind"] in ("forward","adjoint"):
            previous=None
            order=range(self.H) if self.summary["kind"]=="forward" else range(self.H-1,-1,-1)
            for j in order:
                path=self.folder/f"row_{j:03d}.json"
                row=json.loads(path.read_text())
                if (row!=self.summary["rows"][j] or
                    row["method_sha256"]!=self.summary["method_sha256"] or
                    row["parent_record_sha256"]!=previous):
                    raise ValueError("completed recurrence record chain changed")
                previous=digest(path)
            old=ctx.prec;ctx.prec=max(old,256)
            try:
                for key,value_key in (("residual_sequence_upper","local_residual_norm_upper"),
                                      ("terminal_sequence_norm_upper","norm_upper")):
                    computed=sequence_upper([r[value_key] for r in self.summary["rows"]])
                    if self.summary[key]!=computed:
                        raise ValueError("complete aggregate differs from its row bounds")
                if self.summary["hvp_calls"]!=self.H-1:
                    raise ValueError("incomplete HVP count")
            finally:ctx.prec=old

    def row(self,j):
        record=self.summary["rows"][j]
        path=self.folder/record["file"]
        if digest(path)!=record["sha256"]:raise ValueError("input row changed")
        value=np.load(path,allow_pickle=False)
        if value.shape!=(self.d,) or value.dtype!=np.float64 or not np.isfinite(value).all():
            raise ValueError("invalid stored row")
        return value

    def flat_npy_sha256(self):
        header=io.BytesIO()
        np.lib.format.write_array_header_1_0(header,{"descr":np.dtype(np.float64).str,
                           "fortran_order":False,"shape":(self.H*self.d,)})
        result=hashlib.sha256(header.getvalue())
        for j in range(self.H):result.update(self.row(j).tobytes())
        return result.hexdigest()


def store_rows(folder, rows, *, shape, contract, expected_flat_npy_sha256=None):
    """Seal an input stream, checking its canonical flattened .npy identity."""
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    H,d=map(int,shape)
    if H<1 or d<2 or d%2:raise ValueError("positive H and even state dimension required")
    method={"kind":"input_rows","shape":[H,d],"contract":contract,
            "expected_flat_npy_sha256":expected_flat_npy_sha256}
    save_json(folder/"method.json",method)
    header=io.BytesIO()
    np.lib.format.write_array_header_1_0(header,{"descr":np.dtype(np.float64).str,
                                               "fortran_order":False,"shape":(H*d,)})
    hashed=hashlib.sha256(header.getvalue());records=[]
    old=ctx.prec;ctx.prec=max(old,256)
    try:
        for j,value in enumerate(rows):
            value=np.asarray(value,dtype=np.float64)
            if j>=H or value.shape!=(d,) or not np.isfinite(value).all():
                raise ValueError("input stream shape or finiteness failure")
            value=np.ascontiguousarray(value)
            hashed.update(value.tobytes())
            filename=f"row_{j:03d}.npy";save_array(folder/filename,value)
            records.append({"index":j,"file":filename,"sha256":digest(folder/filename),
                            "norm_upper":streaming_norm_upper(value)})
        if len(records)!=H:raise ValueError("short input stream")
        if expected_flat_npy_sha256 is not None and hashed.hexdigest()!=expected_flat_npy_sha256:
            raise ValueError("regenerated probe differs from original sealed Gaussian vector")
        summary={"complete":True,"kind":"input_rows","shape":[H,d],"rows":records,
                 "method_sha256":digest(folder/"method.json"),"flat_npy_sha256":hashed.hexdigest(),
                 "terminal_sequence_norm_upper":sequence_upper([r["norm_upper"] for r in records])}
        save_json(folder/"summary.json",summary)
        return StoredRows(folder)
    finally:ctx.prec=old


def product(folder, inputs, hvp, *, learning_rate, momentum, transpose=False,
            precision_bits=128, contract, max_new_steps=None, progress=None,
            guard=None, storage_fault_hook=None):
    """Resume a causal product. Partial prefixes do not carry a norm bound.

    max_new_steps bounds new transitions, not the number of imported rows.
    storage_fault_hook is a test-only interruption hook after the array write.
    The caller's contract must authenticate the exact operator and HVP code.
    """
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    H,d=inputs.H,inputs.d
    if not isinstance(precision_bits,int) or precision_bits<64:raise ValueError("precision below contract")
    if max_new_steps is not None and (not isinstance(max_new_steps,int) or max_new_steps<0):
        raise ValueError("nonnegative step budget required")
    if not all(math.isfinite(float(x)) for x in (learning_rate,momentum)):
        raise ValueError("nonfinite optimizer constants")
    method={"kind":"adjoint" if transpose else "forward","shape":[H,d],
            "input_manifest_sha256":inputs.identity,"precision_bits":precision_bits,
            "learning_rate":float(learning_rate),"momentum":float(momentum),"contract":contract}
    save_json(folder/"method.json",method);mh=digest(folder/"method.json")
    order=list(range(H-1,-1,-1)) if transpose else list(range(H))
    previous=None;parent=None;records={};new_steps=0
    old=ctx.prec;ctx.prec=precision_bits
    try:
        for j in order:
            if guard is not None:guard()
            rp=folder/f"row_{j:03d}.json";ap=folder/f"row_{j:03d}.npy"
            if rp.exists():
                row=json.loads(rp.read_text())
                if (row["index"]!=j or row["method_sha256"]!=mh or
                    row["parent_record_sha256"]!=parent or row["sha256"]!=digest(ap) or
                    row["input_row_sha256"]!=inputs.summary["rows"][j]["sha256"]):
                    raise ValueError("stored recurrence provenance failure")
                state=np.load(ap,allow_pickle=False)
                if state.shape!=(d,) or state.dtype!=np.float64 or not np.isfinite(state).all():
                    raise ValueError("invalid recurrence row")
                expected_step=(j+1 if transpose else j) if previous is not None else None
                if row["hvp_step"]!=expected_step:raise ValueError("causal index mismatch")
                if not all(math.isfinite(row[k]) and row[k]>=0 for k in
                           ("local_residual_norm_upper","norm_upper")):
                    raise ValueError("invalid local enclosure")
                if previous is None and (row["local_residual_norm_upper"]!=0 or
                                         not np.array_equal(state,inputs.row(j))):
                    raise ValueError("non-exact endpoint copy")
            else:
                if max_new_steps is not None and new_steps>=max_new_steps:
                    return {"complete":False,"completed_steps":len(records),"horizon":H,
                            "new_steps":new_steps,"method_sha256":mh}
                phase=time.perf_counter();injection=inputs.row(j)
                if previous is None:
                    state=injection.copy();error=0.;hvp_step=None
                else:
                    hvp_step=j+1 if transpose else j
                    direction=(adjoint_hvp_direction(previous) if transpose else
                               [arb(float(v)) for v in previous[:d//2]])
                    h=hvp(hvp_step,direction)
                    item=rounded_recurrence_step(previous,injection,h,learning_rate=learning_rate,
                                                 momentum=momentum,transpose=transpose)
                    state=item["next_state"];error=item["local_residual_norm_upper"]
                    del h,direction,item
                ctx.prec=max(256,precision_bits);upper=streaming_norm_upper(state);ctx.prec=precision_bits
                if guard is not None:guard()
                save_array(ap,state)
                if storage_fault_hook is not None:storage_fault_hook(j)
                row={"index":j,"file":ap.name,"sha256":digest(ap),"method_sha256":mh,
                     "parent_record_sha256":parent,"input_row_sha256":inputs.summary["rows"][j]["sha256"],
                     "hvp_step":hvp_step,"local_residual_norm_upper":error,"norm_upper":upper,
                     "seconds":time.perf_counter()-phase}
                save_json(rp,row);new_steps+=1
                if progress is not None:progress(row)
            previous=state;parent=digest(rp);records[j]=row
        ctx.prec=max(256,precision_bits)
        rows=[records[j] for j in range(H)]
        summary={"complete":True,"kind":method["kind"],"shape":[H,d],"method_sha256":mh,
                 "rows":rows,"hvp_calls":H-1,
                 "residual_sequence_upper":sequence_upper([r["local_residual_norm_upper"] for r in rows]),
                 "terminal_sequence_norm_upper":sequence_upper([r["norm_upper"] for r in rows]),
                 "input_manifest_sha256":inputs.identity}
        if guard is not None:guard()
        save_json(folder/"summary.json",summary)
        return summary
    finally:ctx.prec=old
