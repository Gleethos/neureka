package neureka.ndim.config;

import neureka.Neureka;
import neureka.ndim.config.types.ColumnMajorNDConfiguration;
import neureka.ndim.config.types.complex.*;
import neureka.ndim.config.types.simple.*;
import neureka.ndim.config.types.views.SimpleReshapeView;
import neureka.ndim.config.types.virtual.VirtualNDConfiguration;

import java.util.Arrays;
import java.util.Collections;
import java.util.Map;
import java.util.WeakHashMap;

/**
 *  The following is an abstract implementation of the {@link NDConfiguration} which offers a basis for
 *  instantiation and caching of concrete implementations extending this abstract class.
 *  Concrete {@link NDConfiguration} implementations are expected to be immutable which ensures that sharing them is safe.
 *  In order to cash instances based in their field variables, this class comes with a common
 *  implementation of the {@link NDConfiguration#keyCode()} method.
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
    private static final Map<Long, NDConfiguration> _CACHED_NDCS;
    static
    {
        _CACHED_NDCS = Collections.synchronizedMap( new WeakHashMap<>() ) ;
    }

    /**
     *  The following is a global cache for readonly integer arrays.
     *  Warning! This can of course become dangerous when these arrays are being shared and modified.
     *  Please copy them when exposing them to the user.
     */
    private static final Map<Long, int[]> _CACHED_INT_ARRAYS;
    static
    {
        _CACHED_INT_ARRAYS = Collections.synchronizedMap( new WeakHashMap<>() ) ;
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
    protected static int[] _cacheArray( int[] data )
    {
        long key = 0;
        for ( int e : data ) {
            if      ( e <=              10 ) key *=              10;
            else if ( e <=             100 ) key *=             100;
            else if ( e <=           1_000 ) key *=           1_000;
            else if ( e <=          10_000 ) key *=          10_000;
            else if ( e <=         100_000 ) key *=         100_000;
            else if ( e <=       1_000_000 ) key *=       1_000_000;
            else if ( e <=      10_000_000 ) key *=      10_000_000;
            else if ( e <=     100_000_000 ) key *=     100_000_000;
            else if ( e <=   1_000_000_000 ) key *=   1_000_000_000;
            key += Math.abs( e ) + 1;
        }
        int rank = data.length;
        while ( rank != 0 ) {
            rank /= 10;
            key *= 10;
        }
        key += data.length;
        int[] found = _CACHED_INT_ARRAYS.get( key );
        if ( found != null && Arrays.equals(data, found) )
            return found;

        _CACHED_INT_ARRAYS.put(key, data);
        return data;
    }

    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    @Override
    public long keyCode()
    {
        return this.getClass().hashCode() +
               Arrays.hashCode( shape() )       * 1L +
               Arrays.hashCode( translation() ) * 2L +
               Arrays.hashCode( indicesMap() )  * 3L +
               Arrays.hashCode( spread() )      * 4L +
               Arrays.hashCode( offset() )      * 5L +
               ( getLayout() == Layout.ROW_MAJOR ? 0 : 1 );
    }

    @Override
    public boolean equals( NDConfiguration ndc )
    {
        return  this.getClass() == ndc.getClass() &&
                Arrays.equals(shape(),       ndc.shape()      ) &&
                Arrays.equals(translation(), ndc.translation()) &&
                Arrays.equals(indicesMap(),  ndc.indicesMap() ) &&
                Arrays.equals(spread(),      ndc.spread()     ) &&
                Arrays.equals(offset(),      ndc.offset()     ) &&
                this.getLayout() == ndc.getLayout();
    }

    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    public static NDConfiguration construct (
            int[] shape,
            int[] translation,
            int[] indicesMap,
            int[] spread,
            int[] offset,
            Layout layout
    ) {
        for ( int dim : shape ) {
            if ( dim == 0 ) {
                String message = "Trying to create tensor configuration containing shape with dimension 0.\n" +
                        "Shape dimensions must be greater than 0!\n";
                throw new IllegalStateException( message );
            }
        }

        if ( layout == Layout.COLUMN_MAJOR )
            return ColumnMajorNDConfiguration.construct(shape, translation, indicesMap, spread, offset);

        if ( Neureka.get().settings().ndim().isOnlyUsingDefaultNDConfiguration() )
            return ComplexDefaultNDConfiguration.construct(shape, translation, indicesMap, spread, offset);

        boolean isSimple = _isSimpleConfiguration(shape, translation, indicesMap, spread, offset);
        NDConfiguration ndc = null;
        if ( isSimple ) {
            if ( shape.length == 1 ) {
                if ( shape[ 0 ]==1 ) ndc = SimpleScalarConfiguration.construct();
                else ndc = SimpleD1Configuration.construct(shape, translation);
            } else if ( shape.length == 2 ) {
                ndc = SimpleD2Configuration.construct(shape, translation);
            } else if ( shape.length == 3 ) {
                ndc = SimpleD3Configuration.construct(shape, translation);
            } else ndc = SimpleDefaultNDConfiguration.construct(shape, translation);
        } else {
            if ( shape.length == 1 ) {
                if ( shape[ 0 ] == 1 ) ndc = ComplexScalarConfiguration.construct(shape, offset);
                else ndc = ComplexD1Configuration.construct(shape, translation, indicesMap, spread, offset);
            } else if ( shape.length == 2 ) {
                ndc = ComplexD2Configuration.construct(shape, translation, indicesMap, spread, offset);
            } else if ( shape.length == 3 ) {
                ndc = ComplexD3Configuration.construct(shape, translation, indicesMap, spread, offset);
            } else ndc = ComplexDefaultNDConfiguration.construct(shape, translation, indicesMap, spread, offset);
        }
        return ndc;
    }

    protected static <T extends NDConfiguration> NDConfiguration _cached( T ndc )
    {
        assert !( ndc instanceof VirtualNDConfiguration );
        long key = ndc.keyCode();
        NDConfiguration found = _CACHED_NDCS.get( key );
        if ( found != null && ndc.equals(found) ) return found;
        else {
            _CACHED_NDCS.put( key, ndc );
            return ndc;
        }
    }

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

    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    @Override
    public String toString() {
        return "NDConfiguration@"+Integer.toHexString(hashCode())+"#"+Long.toHexString(keyCode())+"[" +
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
                newOffset,
                Layout.ROW_MAJOR
            );
    }

    @Override
    public NDConfiguration newReshaped( int[] newForm )
    {
        //TODO : shape check!
        if ( _isSimpleConfiguration( shape(), translation(), indicesMap(), spread(), offset() ) ) {
            return _simpleReshape( newForm, this );
        } else {
            return new SimpleReshapeView( newForm, this );
            //throw new IllegalStateException("Not ready");
        }


    }


}
