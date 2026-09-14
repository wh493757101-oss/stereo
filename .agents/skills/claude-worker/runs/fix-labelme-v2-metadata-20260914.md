# External worker task

## Objective

Make newly migrated `bz_JSON_v2` annotations open correctly in LabelMe by retaining a null `imageData` key and storing a valid image path relative to each output JSON file.

## Allowed scope

- Inspect `scripts/migrate_labelme_annotations.py`, `tests/test_annotation_migration.py`, and one or two sample files under `datasets/bz_JSON_v2` and `datasets/Rectified_v2`.
- Modify only `scripts/migrate_labelme_annotations.py` and `tests/test_annotation_migration.py`.

## Requirements

- Migrated LabelMe JSON must contain `"imageData": null`; it must not contain embedded base64 image bytes.
- `imagePath`, resolved relative to the output JSON's parent directory, must point to the corresponding image under the supplied `new_image_root` even though annotation and image roots are separate trees.
- Use portable relative paths with forward slashes in JSON, not machine-specific absolute paths.
- Empty negative-frame JSON must follow the same metadata rules.
- Preserve polygon transformation, labels, flags, dimensions, dry-run behavior, and the migration manifest.
- Add focused regression tests for the null key and resolvable cross-root image path.

## Prohibited actions

- Do not modify generated dataset files or any file outside the allowed scope.
- Do not install or update dependencies.
- Do not commit, push, reset, clean, checkout, or switch Git state.
- Do not access external networks or credentials.
- Do not weaken, skip, or delete existing tests.

## Validation

- Run `python -m pytest tests/test_annotation_migration.py -q -W error`.
- If that exact command is unavailable, report it instead of substituting another broad command.

## Return contract

Return a concise summary containing:

- status: completed, blocked, or failed
- files inspected
- files changed
- validation commands and results
- unresolved risks or assumptions
