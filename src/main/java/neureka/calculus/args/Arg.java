package neureka.calculus.args;

import neureka.Component;
import neureka.Tsr;

public abstract class Arg<T> implements Component<Args> {

    private final T _value;

    public Arg(T arg ) { this._value = arg; }

    public T get() { return _value; }

    @Override
    public boolean update(OwnerChangeRequest<Args> changeRequest) {
        return true;
    }


    public static class Derivative<V> extends Arg<Tsr<V>> {
        public Derivative(Tsr<V> arg) { super(arg); }
    }

    public static class Ends extends Arg<int[]> {
        public Ends(int[] arg) { super(arg); }
    }

    public static class VarIdx extends Arg<Integer> {
        public VarIdx(int arg) { super(arg); }
    }

    @Override
    public String toString() {
        if ( _value == null ) return "null";
        else return _value.toString();
    }

}
