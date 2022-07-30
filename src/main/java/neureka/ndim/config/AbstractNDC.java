package neureka.ndim.config;

import neureka.Neureka;
import neureka.common.utility.Cache;
import neureka.ndim.config.types.reshaped.Reshaped1DConfiguration;
import neureka.ndim.config.types.reshaped.Reshaped2DConfiguration;
import neureka.ndim.config.types.reshaped.Reshaped3DConfiguration;
import neureka.ndim.config.types.reshaped.ReshapedNDConfiguration;
import neureka.ndim.config.types.simple.*;
import neureka.ndim.config.types.sliced.*;
import neureka.ndim.config.types.views.SimpleReshapeView;

import java.util.Arrays;

/**
 *  The following is an abstract implementation of the {@link NDConfiguration} which offers a basis for
 *  instantiation and caching of concrete implementations extending this abstract class.
 *  Concrete {@link NDConfiguration} implementations are expected to be immutable which ensures that sharing them is safe.
 *  In order to cash instances based in their field variables, this class comes with a common
 *  implementation of the {@link NDConfiguration#hashCode()} method.
 *  {@link NDConfiguration} implementation instances will be used by tensors which often times
 *  share the same shape, and way of mapping indices to their respective data.
 *  In these cases tensors can simply share their {@link NDConfiguration} instances for memory efficiency.
 */
public abstract class AbstractNDC implements NDConfiguration
{
    /**
     *  Instances implementing the {@link NDConfiguration} interface will be cached in the hashmap below.
     *  In production, we can expect a multitude of tensors having the same shape and also the same way of viewing their data.
     *  Therefore, they will have configuration instances with the same state.
     *  Implementations of {@link NDConfiguration} are expected to be immutable which allows us to have them be
     *  shared between tensors (because they are read only, meaning no side-effects).
     */
    private static final Cache<NDConfiguration> _CACHED_NDCS; // Cached ND-Configurations.
    static
    {
        _CACHED_NDCS = new Cache<>(512);
    }

    /**
     *  The following is a global cache for readonly integer arrays.
     *  Warning! This can of course become dangerous when these arrays are being shared and modified.
     *  Please copy them when exposing them to the user.
     */
    private static final Cache<int[]> _CACHED_INT_ARRAYS; // ND-Configurations are often based on integer arrays representing things like shape, strides, etc...
    static
    {
        _CACHED_INT_ARRAYS = new Cache<>(512);
    }

    /**
     *  This method receives an int array and returns an int array which
     *  can either be the one provided or an array found in the global int array cache residing inside
     *  this class.
     *  Integer array based configurations are not very large,
     *  that is why their state can uniquely be encoded in {@code long} keys.
     *
     * @param data The integer array which ought to be cached.
     * @return The provided array or an already present array found in the int array cache.
     */
    protected static int[] _cacheArray( int[] data ) { return _CACHED_INT_ARRAYS.process( data ); }

    /**
     *   A factory method which creates and {@link NDConfiguration} instances best suited for the
     *   provided raw configuration data...
     */
    static NDConfiguration construct (
            int[] shape,
            int[] translation,
            int[] indicesMap,
            int[] spread,
            int[] offset
    ) {
        for ( int dim : shape )
            if ( dim == 0 )
                throw new IllegalStateException(
                    "Trying to create tensor configuration containing shape with dimension 0.\n" +
                    "Shape dimensions must be greater than 0!\n"
                );

        if ( Neureka.get().settings().ndim().isOnlyUsingDefaultNDConfiguration() )
            return SlicedNDConfiguration.construct(shape, translation, indicesMap, spread, offset);

        boolean isSimple = _isSimpleConfiguration(shape, translation, indicesMap, spread, offset);
        boolean isSimpleTransposed = _isSimpleTransposedConfiguration(shape, spread, offset);

        if ( isSimple )
        {
            if ( shape.length == 1 ) {
                if ( shape[ 0 ] == 1 )
                    return Simple0DConfiguration.construct();
                else
                    return Simple1DConfiguration.construct(shape, translation);
            }
            else if ( shape.length == 2 )
                return Simple2DConfiguration.construct(shape, translation);
            else if ( shape.length == 3 )
                return Simple3DConfiguration.construct(shape, translation);
            else
                return SimpleNDConfiguration.construct(shape, translation);
        }
        if ( isSimpleTransposed )
        {
            if ( shape.length == 1 )
                return Reshaped1DConfiguration.construct(shape, translation, indicesMap);
            else if ( shape.length == 2 )
                return Reshaped2DConfiguration.construct(shape, translation, indicesMap);
            else if ( shape.length == 3 )
                return Reshaped3DConfiguration.construct(shape, translation, indicesMap);
            else
                return ReshapedNDConfiguration.construct(shape, translation, indicesMap);
        }

        if ( shape.length == 1 ) {
            if ( shape[ 0 ] == 1 )
                return Sliced0DConfiguration.construct(shape, offset);
            else
                return Sliced1DConfiguration.construct(shape, translation, indicesMap, spread, offset);
        }
        else if ( shape.length == 2 )
            return Sliced2DConfiguration.construct(shape, translation, indicesMap, spread, offset);
        else if ( shape.length == 3 )
            return Sliced3DConfiguration.construct(shape, translation, indicesMap, spread, offset);

        // This configuration fits every shape:
        return SlicedNDConfiguration.construct(shape, translation, indicesMap, spread, offset);
    }

    protected static <T extends NDConfiguration> T _cached( T ndc ) { return _CACHED_NDCS.process( ndc ); }

    private static boolean _isSimpleConfiguration(
            int[] shape,
            int[] translation,
            int[] indicesMap,
            int[] spread,
            int[] offset
    ) {
        // Note: Column major is not simple because there are no simple column major implementations...
        int[] newTranslation = Layout.ROW_MAJOR.newTranslationFor( shape );
        int[] newSpread = new int[ shape.length ];
        Arrays.fill( newSpread, 1 );
        return  Arrays.equals( translation, newTranslation ) &&
                Arrays.equals( indicesMap, newTranslation ) &&
                Arrays.equals( offset, new int[ shape.length ] ) &&
                Arrays.equals( spread, newSpread );
    }

    private static boolean _isSimpleTransposedConfiguration(
            int[] shape, int[] spread, int[] offset
    ) {
        int[] newSpread = new int[ shape.length ];
        Arrays.fill( newSpread, 1 );
        return Arrays.equals( offset, new int[ shape.length ] ) &&
               Arrays.equals( spread, newSpread );
    }


    @Override
    public final String toString() {
        return "NDConfiguration@"+Integer.toHexString(hashCode())+"#"+Long.toHexString(this.hashCode())+"[" +
                    "layout="+getLayout().name()+","+
                    "shape="+Arrays.toString(shape())+","+
                    "translation="+Arrays.toString(translation())+","+
                    "indicesMap="+Arrays.toString(indicesMap())+","+
                    "spread="+Arrays.toString(spread())+","+
                    "offset="+Arrays.toString(offset())+""+
                "]";
    }

    protected static NDConfiguration _simpleReshape( int[] newForm, NDConfiguration ndc )
    {
        int[] newShape = Utility.rearrange( ndc.shape(), newForm );
        int[] newTranslation = ndc.getLayout().rearrange( ndc.translation(), newShape, newForm );
        int[] newIndicesMap = ndc.getLayout().newTranslationFor( newShape );
        int[] newSpread = new int[ newForm.length ];
        for ( int i = 0; i < newForm.length; i++ ) {
            if ( newForm[ i ] < 0 ) newSpread[ i ] = 1;
            else if ( newForm[ i ] >= 0 ) newSpread[ i ] = ndc.spread( newForm[ i ] );
        }
        int[] newOffset = new int[newForm.length];
        for ( int i = 0; i < newForm.length; i++ ) {
            if ( newForm[ i ] < 0 ) newOffset[ i ] = 0;
            else if ( newForm[ i ] >= 0 ) newOffset[ i ] = ndc.offset( newForm[ i ] );
        }
        return AbstractNDC.construct(
                newShape,
                newTranslation,
                newIndicesMap,
                newSpread,
                newOffset
            );
    }

    @Override
    public NDConfiguration newReshaped( int[] newForm )
    {
        //TODO : shape check!
        if ( _isSimpleConfiguration( shape(), translation(), indicesMap(), spread(), offset() ) )
            return _simpleReshape( newForm, this );
        else
            return new SimpleReshapeView( newForm, this );
    }

    @Override
    public int hashCode() {
        return Long.valueOf(
                   this.getClass().hashCode() +
                   Arrays.hashCode( shape() )       * 1L +
                   Arrays.hashCode( translation() ) * 2L +
                   Arrays.hashCode( indicesMap() )  * 3L +
                   Arrays.hashCode( spread() )      * 4L +
                   Arrays.hashCode( offset() )      * 5L +
                   getLayout().hashCode()
                )
                .hashCode();
    }

    @Override
    public boolean equals( NDConfiguration ndc ) {
        return this.getClass() == ndc.getClass() &&
               Arrays.equals(shape(),       ndc.shape()      ) &&
               Arrays.equals(translation(), ndc.translation()) &&
               Arrays.equals(indicesMap(),  ndc.indicesMap() ) &&
               Arrays.equals(spread(),      ndc.spread()     ) &&
               Arrays.equals(offset(),      ndc.offset()     ) &&
               this.getLayout() == ndc.getLayout();
    }


}
