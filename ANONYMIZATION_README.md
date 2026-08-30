# Anonymous supplement integrity

Text files are path- and author-sanitized for double-blind review. `MANIFEST_SHA256.json` records both the original source hash (the hash cited by the paper and seal chain) and the packaged sanitized hash. A `sanitized` flag identifies every changed payload. Binary artifacts are copied byte-for-byte. Sanitization changes only local paths and author strings; it does not rewrite numerical records, code logic, or source-manifest hashes embedded in the records.
