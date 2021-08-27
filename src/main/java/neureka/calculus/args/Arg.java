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
        public static <V> Derivative<V> of(Tsr<V> arg) { return new Derivative<>(arg); }
        private Derivative(Tsr<V> arg) { super(arg); }
    }

    public static class DerivIdx extends Arg<Integer> {
        public static DerivIdx of( int index ) { return new DerivIdx(index); }
        private DerivIdx(int arg) { super(arg); }
    }

    public static class Ends extends Arg<int[]> {
        public static Ends of( int[] arg ) { return new Ends(arg); }
        private Ends(int[] arg) { super(arg); }
    }

    /**
     *  The following argument is relevant for a particular type of operation, namely: an "indexer". <br>
     *  An indexer automatically applies an operation on all inputs for a given function.
     *  The (indexer) function will execute the sub functions (of the AST) for every input index.
     *  If a particular index is not targeted however this variable will simply default to -1.
     */
    public static class VarIdx extends Arg<Integer> {
        public static VarIdx of( int arg ) { return new VarIdx( arg ); }
        private VarIdx(int arg) { super(arg); }
    }

    @Override
    public String toString() {
        if ( _value == null ) return "null";
        else return _value.toString();
    }

}