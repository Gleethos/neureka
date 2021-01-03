package neureka.utility;

import java.lang.ref.PhantomReference;
import java.lang.ref.ReferenceQueue;
import java.util.ArrayList;
import java.util.List;

public class CustomCleaner implements NeurekaCleaner, Runnable
{
    private final ReferenceQueue<Object> _referenceQueue = new ReferenceQueue<>();
    private final long _timeout = 60*1000;
    private int _registered = 0;

    List<Object> list = new ArrayList<>();

    class ReferenceWithCleanup<T> extends PhantomReference<T>
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
            list.add(new ReferenceWithCleanup<Object>( o, action, _referenceQueue ));
            _registered++;
            if ( _registered == 1 ) new Thread( this::run ).start();
        }
    }

    @Override
    public void run() {
        while ( _registered > 0 ) {
            try {
                ReferenceWithCleanup ref = (ReferenceWithCleanup) _referenceQueue.remove( _timeout );
                if ( ref != null ) {
                    ref.cleanup();
                    _registered--;
                }
            } catch ( Throwable e ) {
                e.printStackTrace();
                // ignore exceptions from the cleanup action
                // (including interruption of cleanup thread)
            }
        }
    }
}
