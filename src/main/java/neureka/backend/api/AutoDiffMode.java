package neureka.backend.api;

public enum AutoDiffMode
{
    FORWARD_ONLY,
    BACKWARD_ONLY,
    FORWARD_AND_BACKWARD,
    NOT_SUPPORTED;

    public boolean allowsForward() {
        return this == FORWARD_AND_BACKWARD || this == FORWARD_ONLY;
    }

    public boolean allowsBackward() {
        return this == FORWARD_AND_BACKWARD || this == BACKWARD_ONLY;
    }
}

