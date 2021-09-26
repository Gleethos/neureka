package neureka.backend.api;

import neureka.Tsr;
import neureka.calculus.args.Arg;
import neureka.calculus.args.Args;
import neureka.devices.Device;

import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

public class Call<D> {

    public static Call.Builder<?> ofTensors( Tsr<?>... tensors ) {
        return new Builder<>(tensors);
    }

    /**
     *  The tensor arguments from which an operation will either
     *  read or to which it will write. <br>
     *  The first entry of this array is usually containing the output tensor,
     *  however this is not a necessity.
     *  Some operation algorithms might use multiple argument entries as output tensors.
     */
    protected Tsr<?>[] _tensors;

    /**
     *  This field references the device on which this ExecutionCall should be executed.
     */
    protected final D _device;


    protected final Args _arguments = new Args();

    protected Call(Tsr<?>[] tensors, D device, List<Arg> arguments ) {
        _tensors = tensors;
        _device = device;
        for ( Arg<?> arg : arguments ) _arguments.set(arg);
    }


    public D getDevice() { return this._device; }

    public Tsr<?>[] getTensors() { return this._tensors; }

    public void mutateTensors( int... indices ) {
        Tsr<?>[] tensors = _tensors.clone();
        for ( int i = 0; i < indices.length; i++ ) {
            _tensors[i] = tensors[indices[i]];
        }
    }

    public <T> Device<T> getDeviceFor(Class<T> supportCheck ) {
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

    public  <V> Tsr<V> getTsrOfType( Class<V> valueTypeClass, int i ) {
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

    public Validator validate() {
        return new Validator();
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


        public <V, D extends Device<V>> Call<D> runningOn(D device ) {
            return new Call<D>(_tensors, device, _arguments.getAll(Arg.class));
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


        public boolean isValid() { return this._isValid; }

        /**
         *  The validity as float being 1.0/true and 0.0/false.
         *
         * @return The current validity of this Validator as float value.
         */
        public float estimation() { return ( this._isValid ? 1.0f : 0.0f ); }


        public Validator first( ExecutionCall.TensorCondition condition ) {
            if ( !condition.check( getTensors()[ 0 ] ) ) this._isValid = false;
            return this;
        }


        public Validator any( ExecutionCall.TensorCondition condition )
        {
            boolean any = false;
            for ( Tsr<?> t : getTensors() ) any = condition.check( t ) || any;
            if ( !any ) this._isValid = false;
            return this;
        }


        public Validator anyNotNull( ExecutionCall.TensorCondition condition )
        {
            boolean any = false;
            for ( Tsr<?> t : getTensors() )
                if ( t != null ) any = condition.check( t ) || any;
            if ( !any ) this._isValid = false;
            return this;
        }


        public Validator all( ExecutionCall.TensorCondition condition )
        {
            boolean all = true;
            for ( Tsr<?> t : getTensors() ) all = condition.check( t ) && all;
            if ( !all ) this._isValid = false;
            return this;
        }


        public Validator allNotNull( ExecutionCall.TensorCondition condition )
        {
            boolean all = true;
            for ( Tsr<?> t : getTensors() )
                if ( t != null ) all = condition.check( t ) && all;
            if ( !all ) this._isValid = false;
            return this;
        }


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
    }

}
