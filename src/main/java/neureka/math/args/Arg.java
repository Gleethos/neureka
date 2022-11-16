package neureka.math.args;

import neureka.common.composition.Component;
import neureka.Tsr;
import neureka.devices.Device;

/**
 *  Extend this class to define additional meta arguments for {@link neureka.math.Functions}.
 *  More complex types of operations need additional parameters/arguments.
 *  The {@link neureka.backend.main.operations.other.Randomization}
 *  operation for example receives the {@link Seed} argument as a basis
 *  for deterministic pseudo random number generation...
 *
 * @param <T> The type parameter defining the type of argument.
 */
public abstract class Arg<T> implements Component<Args> {

    private final T _value;

    public Arg( T arg ) { _value = arg; }

    public T get() {
        if ( _value instanceof int[]     ) return (T) ((int[])    _value).clone();
        if ( _value instanceof float[]   ) return (T) ((float[])  _value).clone();
        if ( _value instanceof double[]  ) return (T) ((double[]) _value).clone();
        if ( _value instanceof long[]    ) return (T) ((long[])   _value).clone();
        if ( _value instanceof short[]   ) return (T) ((short[])  _value).clone();
        if ( _value instanceof byte[]    ) return (T) ((byte[])   _value).clone();
        if ( _value instanceof char[]    ) return (T) ((char[])   _value).clone();
        if ( _value instanceof boolean[] ) return (T) ((boolean[])_value).clone();
        return _value;
    }

    @Override
    public boolean update(OwnerChangeRequest<Args> changeRequest) { return true; }


    public static class Derivative<V> extends Arg<Tsr<V>> {
        public static <V> Derivative<V> of(Tsr<V> arg) { return new Derivative<>(arg); }
        private Derivative(Tsr<V> arg) { super(arg); }
    }

    /**
     * This is an import argument whose
     * role might not be clear at first :
     * An operation can have multiple inputs, however
     * when calculating the derivative for a forward or backward pass
     * then one must know which derivative ought to be calculated.
     * So the "derivative index" targets said input.
     * This property is -1 when no derivative should be calculated,
     * however 0... when targeting an input to calculate the derivative of.
     */
    public static class DerivIdx extends Arg<Integer> {
        public static DerivIdx of( int index ) { return new DerivIdx(index); }
        private DerivIdx(int arg) { super(arg); }
    }

    public static class Axis extends Arg<Integer> {
        public static Axis of(int index ) { return new Axis(index); }
        private Axis(int arg) { super(arg); }
    }

    public static class Ends extends Arg<int[]> {
        public static Ends of( int[] arg ) { return new Ends(arg); }
        private Ends(int[] arg) { super(arg); }
    }

    public static class TargetDevice extends Arg<Device<?>> {
        public static TargetDevice of( Device<?> arg ) { return new TargetDevice(arg); }
        private TargetDevice(Device<?> arg) { super(arg); }
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

    public static class MinRank extends Arg<Integer> {
        public static MinRank of( int arg ) { return new MinRank( arg ); }
        private MinRank( int arg ) { super(arg); }
    }

    public static class Seed extends Arg<Long> {
        public static Seed of( long arg ) { return new Seed( arg ); }
        private Seed( long arg ) { super(arg); }
    }

    public static class Shape extends Arg<int[]> {
        public static Shape of( int[] arg ) { return new Shape( arg ); }
        private Shape( int[] arg ) { super(arg); }
    }

    public static class Offset extends Arg<int[]> {
        public static Offset of( int[] arg ) { return new Offset( arg ); }
        private Offset( int[] arg ) { super(arg); }
    }

    public static class Stride extends Arg<int[]> {
        public static Stride of( int[] arg ) { return new Stride( arg ); }
        private Stride( int[] arg ) { super(arg); }
    }

    @Override
    public String toString() { return this.getClass().getSimpleName() + "[" + _value + "]"; }

}
