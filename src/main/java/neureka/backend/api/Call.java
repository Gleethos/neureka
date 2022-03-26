package neureka.backend.api;

import neureka.Tsr;
import neureka.backend.api.algorithms.fun.SuitabilityPredicate;
import neureka.calculus.args.Arg;
import neureka.calculus.args.Args;
import neureka.devices.Device;

import java.util.Arrays;
import java.util.List;
import java.util.Objects;
import java.util.function.Function;
import java.util.stream.Collectors;

/**
 *  Instances of this class model simple execution calls to the backend.
 *  They can be passed to {@link neureka.calculus.Function} instances in order to get full
 *  control over the execution via the use of call {@link Args}.
 *  This class is the precursor class of {@link ExecutionCall} which is a more complete
 *  execution state bundle used inside the backend.
 *
 * @param <D> The type parameter which defines the {@link Device} targeted by this {@link Call}.
 */
public class Call<D>
{

    public static <V, T extends Device<V>> Call.Builder<V,T> to( T device ) { return new Builder<V,T>( device ); }

    /**
     *  The tensor arguments from which an operation will either
     *  read or to which it will write. <br>
     *  The first entry of this array is usually containing the output tensor,
     *  however this is not a necessity.
     *  Some operation algorithms might use multiple argument entries as output tensors.
     */
    protected final Tsr<?>[] _tensors;
    /**
     *  This field references the device on which this ExecutionCall should be executed.
     */
    protected final D _device;
    /**
     *  Meta arguments which are usually specific to certain operations.
     */
    protected final Args _arguments = new Args();


    protected Call( Tsr<?>[] tensors, D device, List<Arg> arguments ) {
        _tensors = tensors.clone();
        _device = device;
        for ( Arg<?> arg : arguments ) _arguments.set(arg);
    }

    /**
     * @return The device targeted by this call for execution.
     */
    public D getDevice() { return _device; }

    /**
     * @return The {@link Tsr} parameters of this {@link Call} for execution.
     */
    public Tsr<?>[] inputs() { return _tensors.clone(); }

    /**
     * @return The number of input tensors.
     */
    public int arity() { return _tensors.length; }

    /**
     * @param i The index of the tensor argument which should be returned.
     * @return The {@code i}'th {@link Tsr} parameter of this {@link Call} for execution.
     */
    public Tsr<?> input( int i ) {
        return _tensors[ i ];
    }

    public void rearrangeInputs( int... indices ) {
        Tsr<?>[] tensors = _tensors.clone();
        for ( int i = 0; i < indices.length; i++ ) {
            _tensors[i] = tensors[indices[i]];
        }
    }

    public <T> Device<T> getDeviceFor( Class<T> supportCheck ) {
        // TODO: Make it possible to query device for type support!
        return (Device<T>) this.getDevice();
    }

    public List<Arg> allMetaArgs() {
        return _arguments.getAll(Arg.class).stream().map( a -> (Arg<Object>) a ).collect(Collectors.toList());
    }

    public <V, T extends Arg<V>> T get( Class<T> argumentClass ) {
        return _arguments.get(argumentClass);
    }

    public <V, T extends Arg<V>> V getValOf( Class<T> argumentClass ) {
        return _arguments.valOf(argumentClass);
    }

    public int getDerivativeIndex() {
        return this.getValOf( Arg.DerivIdx.class );
    }

    public  <V> Tsr<V> input( Class<V> valueTypeClass, int i ) {
        Tsr<?>[] tensors = _tensors;
        if ( valueTypeClass == null ) {
            throw new IllegalArgumentException(
                    "The provided tensor type class is null!\n" +
                            "Type safe access to the tensor parameter at index '"+i+"' failed."
            );
        }
        if ( tensors[ i ] != null ) {
            Class<?> tensorTypeClass = tensors[ i ].getValueClass();
            if ( !valueTypeClass.isAssignableFrom(tensorTypeClass) ) {
                throw new IllegalArgumentException(
                    "The item value type of the tensor stored at parameter position '"+i+"' is " +
                    "'"+tensorTypeClass.getSimpleName()+"' and is not a sub-type of the provided " +
                    "type '"+valueTypeClass.getSimpleName()+"'."
                );
            }
        }
        return (Tsr<V>) tensors[ i ];
    }

    public Validator validate() { return new Validator(); }

    public static class Builder<V, T extends Device<V>>
    {
        private final T _device;
        private Tsr<V>[] _tensors;
        private final Args _arguments = Args.of( Arg.DerivIdx.of(-1), Arg.VarIdx.of(-1) );

        private Builder( T device ) { _device = device; }

        @SafeVarargs
        public final <N extends V> Builder<V,T> with( Tsr<N>... tensors ) {
            _tensors = (Tsr<V>[]) tensors;
            return this;
        }

        public Builder<V,T> andArgs( List<Arg> arguments ) {
            for ( Arg<?> argument : arguments ) _arguments.set(argument);
            return this;
        }

        public Builder<V,T> andArgs( Arg<?>... arguments ) {
            return andArgs(Arrays.stream(arguments).collect(Collectors.toList()));
        }

        public Call<T> get() {
            return new Call<T>( _tensors, _device, _arguments.getAll( Arg.class ) );
        }

    }


    /**
     *  This is a simple nested class offering various lambda based methods
     *  for validating the tensor arguments stored inside this {@link ExecutionCall}.
     *  It is a useful tool readable as well as concise validation of a given
     *  request for execution, that is primarily used inside implementations of the middle
     *  layer of the backend-API architecture ({@link Algorithm#isSuitableFor(ExecutionCall)}).
     */
    public class Validator {

        private boolean _isValid = true;


        public boolean isValid() { return _isValid; }

        /**
         *  The validity as float being &#62;0/true and 0/false.
         *  If the {@link Call} is valid then a suitability estimation of 0.9f
         *  will be returned simply because a suitability of 1 would mean
         *  that no other algorithm could ever compete with this one if if was
         *  faster or simply better suited!
         *
         * @return The current validity of this Validator as float value.
         */
        public float basicSuitability() {
            return suitabilityIfValid( SuitabilityPredicate.GOOD );
        }

        public float suitabilityIfValid(float estimationIfValid ) {
            return ( _isValid ? estimationIfValid : SuitabilityPredicate.UNSUITABLE );
        }

        public Estimator getEstimator() { return new Estimator( _isValid ); }

        public Validator first( TensorCondition condition ) {
            if ( _isValid && !condition.check( input( 0 ) ) ) _isValid = false;
            return this;
        }

        public Validator last( TensorCondition condition ) {
            if ( _isValid && !condition.check( input( arity() - 1 ) ) ) _isValid = false;
            return this;
        }

        public Validator tensors( TensorsCondition condition ) {
            if ( _isValid && !condition.check( _tensors ) ) _isValid = false;
            return this;
        }

        public Validator any( TensorCondition condition )
        {
            if ( _isValid && !_anyMatch( condition ) ) _isValid = false;
            return this;
        }

        private boolean _anyMatch( TensorCondition condition )
        {
            boolean any = false;
            for ( Tsr<?> t : _tensors ) any = condition.check( t ) || any;
            return any;
        }

        public Validator anyNotNull( TensorCondition condition )
        {
            if ( !_anyNotNullMatch( condition ) ) _isValid = false;
            return this;
        }

        private boolean _anyNotNullMatch( TensorCondition condition )
        {
            boolean any = false;
            for ( Tsr<?> t : _tensors )
                if ( t != null ) any = condition.check( t ) || any;
            return any;
        }

        public Validator all( TensorCondition condition )
        {
            if ( !_allMatch( condition ) ) _isValid = false;
            return this;
        }

        public Validator allNotNullHaveSame( TensorProperty propertySource ) {
            if ( !_allHaveSame( propertySource ) ) _isValid = false;
            return this;
        }

        private boolean _allHaveSame( TensorProperty propertySource ) {
            Object last = null;
            boolean firstWasSet = false;
            for ( Tsr<?> t : inputs() ) {
                if ( t != null ) {
                    Object current = propertySource.propertyOf(t);
                    if ( !Objects.equals(last, current) && firstWasSet )
                        return false;
                    last = current; // Note: shapes are cached!
                    firstWasSet = true;
                }
            }
            return true;
        }

        private boolean _allMatch( TensorCondition condition ) {
            boolean all = true;
            for ( Tsr<?> t : _tensors ) all = condition.check( t ) && all;
            return all;
        }

        public Validator allNotNull( TensorCondition condition )
        {
            if ( _isValid && !_allNotNullMatch( condition ) ) _isValid = false;
            return this;
        }

        private boolean _allNotNullMatch( TensorCondition condition )
        {
            boolean all = true;
            for ( Tsr<?> t : _tensors )
                if ( t != null ) all = condition.check( t ) && all;
            return all;
        }

        public Validator all( TensorCompare compare )
        {
            if ( _isValid && !_allMatch( compare ) ) _isValid = false;
            return this;
        }

        private boolean _allMatch( TensorCompare compare ) {
            boolean all = true;
            Tsr<?> last = null;
            for ( Tsr<?> current : _tensors ) {
                if ( last != null && !compare.check( last, current ) ) all = false;
                last = current; // Note: shapes are cached!
            }
            return all;
        }

        public <T> Validator allShare( Function<Tsr<?>, T> propertyProvider ) {
            T first = null;
            for ( Tsr<?> t : _tensors ) {
                if ( t != null ) {
                    T found = propertyProvider.apply( t );
                    if ( first == null && found != null ) first = found;
                    else if ( first != null ) {
                        if ( !first.equals(found) ) {
                            _isValid = false;
                            return this;
                        }
                    }
                }
            }
            return this;
        }

        public class Estimator {

            private float _estimation;

            public Estimator( boolean isValid ) {
                _estimation = ( isValid ? SuitabilityPredicate.OKAY : SuitabilityPredicate.UNSUITABLE );
            }

            private void _mod( float f ) {
                f = Math.max( -1f, f );
                f = Math.min(  1f, f );
                _estimation *= ( 1 + ( f * ( 1 - _estimation ) ) );
            }

            public Estimator goodIfAll( TensorCondition condition ) { if ( _allMatch( condition ) ) _mod(0.5f); return this; }

            public Estimator badIfAll( TensorCondition condition ) { if ( _allMatch( condition ) ) _mod(-0.5f); return this; }

            public Estimator goodIfAnyNonNull( TensorCondition condition ) { return goodIfAny( t -> t != null && condition.check(t) ); }

            public Estimator goodIfAny( TensorCondition condition ) { if ( _anyMatch( condition ) ) _mod(0.5f); return this; }

            public Estimator badIfAnyNonNull( TensorCondition condition ) { return badIfAny( t -> t != null && condition.check(t) ); }

            public Estimator badIfAny( TensorCondition condition ) { if ( _anyMatch( condition ) ) _mod(-0.5f); return this; }

            public Estimator goodIfAll( TensorCompare condition ) { if ( _allMatch( condition ) ) _mod(0.5f); return this; }

            public Estimator badIfAll( TensorCompare condition ) { if ( _allMatch( condition ) ) _mod(-0.5f); return this; }

            public float getEstimation() { return _estimation; }
        }

    }

    public interface TensorProperty     { Object  propertyOf( Tsr<?> tensor ); }
    public interface TensorCompare      { boolean check( Tsr<?> first, Tsr<?> second ); }
    public interface TensorsCondition   { boolean check( Tsr<?>[] tensors ); }
    public interface TensorCondition    { boolean check( Tsr<?> tensor ); }
    public interface DeviceCondition    { boolean check( Device<?> device ); }
    public interface OperationCondition { boolean check( Operation type ); }

}
