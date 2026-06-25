# Task Log Archives

Use this folder for archived task history when the active `TASK_LOG.md` becomes hard to scan.

## Naming Convention
Use one of these patterns:
- `TASK_LOG_YYYY_Q1.md`
- `TASK_LOG_YYYY_Q2.md`
- `TASK_LOG_YYYY_Q3.md`
- `TASK_LOG_YYYY_Q4.md`
- `TASK_LOG_<phase>.md`

Quarter-based naming is the default. Phase-based naming is fine when the project has clear stages and quarter splits would be less useful.

## Archive Trigger Rule
Archive older entries when the active `TASK_LOG.md` becomes hard to scan during normal work.

Signals that it is time to archive:
- important recent entries are getting buried,
- the file is slow to review at task start,
- the log contains multiple completed phases that no longer need to stay in the active file.

## Archive Rules
- Keep `TASK_LOG.md` as the active working log.
- Move only older completed entries into archive files.
- Preserve chronological order inside each archive file.
- Leave recent or currently relevant entries in the active log.
- Add a short note in `TASK_LOG.md` if a large archive move would otherwise confuse future readers.
