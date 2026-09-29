"""Wall motion errors shared by coupling geometry and VPM integration."""


class WallCorrectionTooLargeError(RuntimeError):
    """A wall crossing requires a smaller particle integration interval."""
