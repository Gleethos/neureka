package neureka.devices;

import java.lang.ref.PhantomReference;
import java.lang.ref.ReferenceQueue;
import java.util.ArrayList;
import java.util.List;

/**
 *  This class stores actions which are being executed when an associated object is being garbage collected.
 *  This class is similar to the cleaner class introduced in JDK 11, however the minimal version compatibility target
 *  for Neureka is Java 8, which means that this cleaner class introduced in Java 11 is not available here!
 *  That is why a custom cleaner implementation is being defined below.<br>
 */
final class CustomDeviceCleaner implements DeviceCleaner, Runnable
{
    private final ReferenceQueue<Object> _referenceQueue = new ReferenceQueue<>();
    private final long _timeout = 60 * 1000;
    private int _registered = 0;

    List<Object> list = new ArrayList<>();

    static class ReferenceWithCleanup<T> extends PhantomReference<T>
    {
        private Runnable _action;

        ReferenceWithCleanup(T o, Runnable action, ReferenceQueue<T> queue) {
            super( o, queue );
            _action = action;
        }
        public void cleanup() {
            _action.run();
        }
    }

    @Override
    public void register(Object o, Runnable action) {
        synchronized ( _referenceQueue ) {
            list.add(new ReferenceWithCleanup<Object>(o, action, _referenceQueue));
            _registered++;
            if ( _registered == 1 ) new Thread( this::run ).start();
        }
    }

    @Override
    public void run() {
        while ( _registered > 0 ) {
            try {
                ReferenceWithCleanup ref = (ReferenceWithCleanup) _referenceQueue.remove(_timeout);
                if ( ref != null ) {
                    try {
                        ref.cleanup();
                    } catch ( Throwable e ) {
                        e.printStackTrace();
                        // ignore exceptions from the cleanup action
                        // (including interruption of cleanup thread)
                    }
                    _registered--;
                }
            } catch ( Throwable e ) {
                e.printStackTrace(); // The queue failed
            }
        }
    }

    @Override
    public String toString() {
        return this.getClass().getSimpleName()+"@"+Integer.toHexString(this.hashCode())+"[" +
                    "registered=" + _registered +
                "]";
    }

}
