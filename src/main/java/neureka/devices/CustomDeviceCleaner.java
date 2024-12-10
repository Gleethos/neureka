package neureka.devices;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.lang.ref.PhantomReference;
import java.lang.ref.ReferenceQueue;
import java.util.ArrayList;
import java.util.List;

/**
 *  This class stores actions which are being executed when an associated object is being garbage collected.
 *  This class is similar to the cleaner class introduced in JDK 11, however the minimal version compatibility target
 *  for Neureka is Java 8, which means that this cleaner class introduced in Java 11 is not available here!
 *  That is why a custom cleaner implementation is being defined below.<br>
 *  <br> <br>
 *  <b>Warning: This is an internal class, meaning it should not be used
 *  anywhere but within this library. <br>
 *  This class or its public methods might change or get removed in future versions!</b>
 */
final class CustomDeviceCleaner implements DeviceCleaner
{
    private static final Logger log = LoggerFactory.getLogger(CustomDeviceCleaner.class);
    private static final CustomDeviceCleaner _INSTANCE = new CustomDeviceCleaner();
    private static final long _QUEUE_TIMEOUT = 60 * 1000;

    private final ReferenceQueue<Object> _referenceQueue = new ReferenceQueue<>();
    private final List<ReferenceWithCleanup<Object>> _toBeCleaned = new ArrayList<>();
    private final Thread _thread;


    public static CustomDeviceCleaner getInstance() {
        return _INSTANCE;
    }

    CustomDeviceCleaner() {
        _thread = new Thread(this::run, "Neureka-Cleaner");
    }


    static class ReferenceWithCleanup<T> extends PhantomReference<T>
    {
        private Runnable _action;

        ReferenceWithCleanup( T o, Runnable action, ReferenceQueue<T> queue ) {
            super( o, queue );
            _action = action;
        }
        public void cleanup() {
            if ( _action != null ) {
                try {
                    _action.run();
                } catch (Exception e) {
                    log.error("Failed to execute cleanup action '"+_action+"'.", e);
                } finally {
                    _action = null;
                }
            }
        }
    }

    public void register( Object o, Runnable action ) {
        if ( o == null ) {
            log.warn("Attempt to register a null object for cleanup. This is not allowed!");
            try {
                action.run();
            } catch (Exception e) {
                log.error("Failed to execute cleanup action '"+action+"'.", e);
            }
            return;
        }
        synchronized ( _referenceQueue ) {
            _toBeCleaned.add(new ReferenceWithCleanup<>(o, action, _referenceQueue));
            if ( _toBeCleaned.size() == 1 ) {
                if ( !_thread.isAlive() ) {
                    _thread.start();
                }
                else {
                    // We notify the cleaner thread that there are new items to be cleaned
                    synchronized ( _thread ) {
                        _thread.notify();
                    }
                }
            }
        }
    }

    private void run() {
        if ( !_thread.isAlive() ) {
            _thread.start();
        }
        while ( _thread.isAlive() ) {
            while ( !_toBeCleaned.isEmpty() ) {
                checkCleanup();
            }
            try {
                synchronized ( _thread ) {
                    _thread.wait();
                }
            } catch (Exception e) {
                log.error("Failed to make cleaner thread wait for cleaning notification!", e);
            }
        }
    }

    private void checkCleanup() {
        try {
            ReferenceWithCleanup<Object> ref = (ReferenceWithCleanup<Object>) _referenceQueue.remove(_QUEUE_TIMEOUT);
            if ( ref != null ) {
                try {
                    ref.cleanup();
                } catch ( Throwable e ) {
                    log.error("Failed to perform cleanup!", e);
                } finally {
                    _toBeCleaned.remove(ref);
                }
            }
        } catch ( Throwable e ) {
            log.error("Failed to call 'remove()' on cleaner internal queue.", e);
        }
    }

    @Override
    public String toString() {
        return this.getClass().getSimpleName()+"@"+Integer.toHexString(this.hashCode())+"[" +
                    "registered=" + _toBeCleaned.size() +
                "]";
    }

}
