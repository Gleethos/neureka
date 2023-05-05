package neureka.devices;

import java.util.Objects;
import java.util.function.Consumer;

public final class ReferenceCounter
{
    public enum ChangeType { INCREMENT, DECREMENT, FULL_DELETE }
    public static class ChangeEvent {
        private final ChangeType changeType;
        private final int change;
        private final int count;

        public ChangeEvent(ChangeType changeType, int change, int count ) {
            this.changeType = changeType;
            this.change = change;
            this.count = count;
        }

        public ChangeType type() { return changeType; }

        public int change() { return change; }

        public int currentCount() { return count; }
    }

    private int _count = 0;
    private final Consumer<ChangeEvent> _action;


    public ReferenceCounter( Consumer<ChangeEvent> action ) { _action = Objects.requireNonNull(action); }

    public void increment() {
        if ( _count < 0 ) throw new IllegalStateException("Cannot increment a reference counter with a negative count!");
        _count++;
        _action.accept(new ChangeEvent(ChangeType.INCREMENT, 1, _count));
    }

    public void decrement() {
        if ( _count == 0 ) throw new IllegalStateException("Cannot decrement a reference counter with a count of zero!");
        _count--;
        _action.accept(new ChangeEvent(ChangeType.DECREMENT, -1, _count));
    }

    public void fullDelete() {
        if ( _count == 0 ) return; // Cleanup action already performed by decrement()!
        if ( _count < 0 ) throw new IllegalStateException("Cannot decrement a reference counter with a negative count!");
        _action.accept(new ChangeEvent(ChangeType.FULL_DELETE, -_count, 0));
        _count = 0;
    }

    public int count() { return _count; }
}
