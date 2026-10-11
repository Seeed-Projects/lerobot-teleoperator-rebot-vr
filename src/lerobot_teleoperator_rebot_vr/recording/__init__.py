"""Local demonstration recording, independent of motion tuning and CSV logging.

Boundaries:
  config: camera/recording settings and local dataset naming.
  buttons: left-menu events from raw tracking; no motion button remapping.
  cameras: USB capture threads and timestamp/age/skew validation.
  dataset: the only LeRobot-dependent storage adapter (v3, degrees, RGB video).
  session: episode state, sampling and bounded queues; runtime observer hooks.
  cli: RS-only validation, hardware preflight and resource ownership.

Lifecycle: open cameras/storage BEFORE ArmRuntime; on_ready excludes startup
motion; on_step samples feedback + final sent action; on_stop rejects unfinished
episodes BEFORE exit motion; close joins IO AFTER the robot has shut down.
Disk work belongs to the writer thread, never an observer hook. Native video
queues are also bounded; any dropped frames invalidate the recording.
"""
