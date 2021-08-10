package neureka.backend.api;

import neureka.Component;
import neureka.Tsr;

public abstract class Argument<T> implements Component<Args> {

    private final T _value;

    public Argument( T arg ) { this._value = arg; }

    public T get() { return _value; }

    @Override
    public boolean update(OwnerChangeRequest<Args> changeRequest) {
        return true;
    }


    public static class Derivative<V> extends Argument<Tsr<V>> {
        public Derivative(Tsr<V> arg) { super(arg); }
    }

    public static class Ends extends Argument<int[]> {
        public Ends(int[] arg) { super(arg); }
    }

}
