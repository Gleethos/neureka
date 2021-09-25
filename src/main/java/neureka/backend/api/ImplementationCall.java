package neureka.backend.api;

import neureka.Tsr;
import neureka.calculus.args.Arg;
import neureka.calculus.args.Args;
import neureka.devices.Device;

import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.Stream;

public interface ImplementationCall<D> {

    D getDevice();

    default <T> Device<T> getDeviceFor(Class<T> supportCheck ) {
        // TODO: Make it possible to query device for type support!
        return (Device<T>) this.getDevice();
    }

    public Tsr<?>[] getTensors();

    <V, T extends Arg<V>> V getValOf( Class<T> argumentClass );

    <V, T extends Arg<V>> T get( Class<T> argumentClass );

    default <V> Tsr<V> getTsrOfType( Class<V> valueTypeClass, int i ) {
        Tsr<?>[] tensors = this.getTensors();
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

    default Validator validate() {
        return new Validator()
        {
            private boolean _isValid = true;

            @Override
            public boolean isValid() { return this._isValid; }

            /**
             *  The validity as float being 1.0/true and 0.0/false.
             *
             * @return The current validity of this Validator as float value.
             */
            @Override
            public float estimation() { return ( this._isValid ? 1.0f : 0.0f ); }

            @Override
            public Validator first( ExecutionCall.TensorCondition condition ) {
                if ( !condition.check( getTensors()[ 0 ] ) ) this._isValid = false;
                return this;
            }

            @Override
            public Validator any( ExecutionCall.TensorCondition condition )
            {
                boolean any = false;
                for ( Tsr<?> t : getTensors() ) any = condition.check( t ) || any;
                if ( !any ) this._isValid = false;
                return this;
            }

            @Override
            public Validator anyNotNull( ExecutionCall.TensorCondition condition )
            {
                boolean any = false;
                for ( Tsr<?> t : getTensors() )
                    if ( t != null ) any = condition.check( t ) || any;
                if ( !any ) this._isValid = false;
                return this;
            }

            @Override
            public Validator all( ExecutionCall.TensorCondition condition )
            {
                boolean all = true;
                for ( Tsr<?> t : getTensors() ) all = condition.check( t ) && all;
                if ( !all ) this._isValid = false;
                return this;
            }

            @Override
            public Validator allNotNull( ExecutionCall.TensorCondition condition )
            {
                boolean all = true;
                for ( Tsr<?> t : getTensors() )
                    if ( t != null ) all = condition.check( t ) && all;
                if ( !all ) this._isValid = false;
                return this;
            }

            @Override
            public Validator all( ExecutionCall.TensorCompare compare )
            {
                boolean all = true;
                Tsr<?> last = null;
                for ( Tsr<?> current : getTensors() ) {
                    if ( last != null && !compare.check( last, current ) ) all = false;
                    last = current; // Note: shapes are cached!
                }
                if ( !all ) this._isValid = false;
                return this;
            }
        };
    }

    public static class Builder<D>
    {
        private Tsr<?>[] _tensors;
        private final Args _arguments = Args.of(
                                                    Arg.DerivIdx.of(-1),
                                                    Arg.VarIdx.of(-1)
                                            );

        private Builder(Tsr<?>[] tensors) { _tensors = tensors; }

        public Builder<D> andArgs( List<Arg> arguments ) {
            for ( Arg argument : arguments ) _arguments.set(argument);
            return this;
        }

        public Builder<D> andArgs( Arg<?>... arguments ) {
            return andArgs(Arrays.stream(arguments).collect(Collectors.toList()));
        }


        public <V, D extends Device<V>> ImplementationCall<D> runningOn( D device ) {
            return new ImplementationCall<D>() {
                @Override public D getDevice() { return device; }
                @Override public Tsr<?>[] getTensors() { return _tensors; }
                @Override
                public <V, T extends Arg<V>> V getValOf( Class<T> argumentClass ) {
                    return _arguments.valOf( argumentClass );
                }
                @Override
                public <V, T extends Arg<V>> T get(Class<T> argumentClass) {
                    return _arguments.get( argumentClass );
                }
            };
        }
    }


    /**
     *  This is a simple nested class offering various lambda based methods
     *  for validating the tensor arguments stored inside this {@link ExecutionCall}.
     *  It is a useful tool readable as well as concise validation of a given
     *  request for execution, that is primarily used inside implementations of the middle
     *  layer of the backend-API architecture ({@link Algorithm#isSuitableFor(ExecutionCall)}).
     */
    public interface Validator {
        boolean isValid();
        default float estimation() { return ( this.isValid() ) ? 1.0f : 0.0f; }

        Validator first( ExecutionCall.TensorCondition condition );

        Validator any( ExecutionCall.TensorCondition condition );

        Validator anyNotNull( ExecutionCall.TensorCondition condition );

        Validator all( ExecutionCall.TensorCondition condition );

        Validator allNotNull( ExecutionCall.TensorCondition condition );

        Validator all( ExecutionCall.TensorCompare compare );
    }

}
