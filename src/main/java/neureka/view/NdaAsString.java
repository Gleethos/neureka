/*
MIT License

Copyright (c) 2019 Gleethos

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

      _______                   _____ _        _
     |__   __|       /\        / ____| |      (_)
        | |___ _ __ /  \   ___| (___ | |_ _ __ _ _ __   __ _
        | / __| '__/ /\ \ / __|\___ \| __| '__| | '_ \ / _` |
        | \__ \ | / ____ \\__ \____) | |_| |  | | | | | (_| |
        |_|___/_|/_/    \_\___/_____/ \__|_|  |_|_| |_|\__, |
                                                        __/ |
                                                       |___/
        A utility class for tensor stringification !

*/

package neureka.view;

import neureka.Nda;
import neureka.Neureka;
import neureka.Tsr;
import neureka.autograd.GraphNode;
import neureka.framing.NDFrame;
import neureka.ndim.config.NDConfiguration;

import java.text.DecimalFormat;
import java.text.DecimalFormatSymbols;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Locale;
import java.util.function.Function;

/**
 *  This class is in essence a simple wrapper class for a tensor and a StringBuilder
 *  Methods in this class use the builder in order to construct a String representation
 *  for said tensor.
 *  These methods perform the String building based on a set of certain
 *  configurations provided by the {@link NDPrintSettings}!
 */
public final class NdaAsString
{
    private final int     _cellSize;
    private final int     _rowLimit;
    private final boolean _hasGradient;
    private final boolean _isCompact;
    private final boolean _isMultiline;
    private final boolean _haveSlimNumbers;
    private final boolean _hasValue;
    private final boolean _hasShape;
    private final boolean _hasRecursiveGraph;
    private final boolean _hasDerivatives;
    private final boolean _isCellBound;
    private final String  _prefix;
    private final String  _postfix;
    private final String  _indent;

    private final int[]   _shape;
    private final Tsr<?>  _tensor;
    private final boolean _legacy;

    NDPrintSettings _config;
    private StringBuilder _asStr;

    /**
     *  A builder providing multiple different configuration options for building
     *  a {@link NdaAsString} instance in a fluent way.
     */
    public static Builder representing( Nda<?> t ) {
        return new Builder() {
            /**
             * @param configMap The configuration map used as basis for turning the wrapped {@link Tsr} to a {@link String}.
             * @return A new {@link NdaAsString} based on the provided configuration.
             */
            @Override
            public NdaAsString withConfig(NDPrintSettings configMap ) {
                return new NdaAsString( t, configMap );
            }
            /**
             * @param config The configuration used as basis for turning the wrapped {@link Tsr} to a {@link String}.
             * @return A new {@link NdaAsString} based on the provided configuration.
             */
            @Override
            public NdaAsString withConfig(String config ) { return withConfig( Neureka.get().settings().view().getNDPrintSettings().clone().with(config) ); }
            /**
             * @return A new {@link NdaAsString} based on the default configuration.
             */
            @Override
            public NdaAsString byDefaults() { return withConfig( Neureka.get().settings().view().getNDPrintSettings() ); }
        };
    }

    private NdaAsString( Nda<?> tensor, NDPrintSettings settings )
    {
        if ( tensor.getNDConf() != null )
            _shape = tensor.getNDConf().shape();
        else
            _shape = new int[0];

        _config = settings.clone();
        _tensor = (Tsr<?>) tensor; // There is only one implementation for both Tsr and Nda...

        _isCompact          = _config.getIsScientific() ;
        _rowLimit           = _config.getRowLimit() ;
        _cellSize           = _config.getCellSize() ;
        _hasGradient        = _config.getHasGradient() ;
        _isMultiline        = _config.getIsMultiline() ;
        _haveSlimNumbers    = _config.getHasSlimNumbers() ;
        _hasValue           = _config.getHasValue() ;
        _hasShape           = _config.getHasShape() ;
        _hasRecursiveGraph  = _config.getHasRecursiveGraph() ;
        _hasDerivatives     = _config.getHasDerivatives() ;
        _isCellBound        = _config.getIsCellBound() ;
        _postfix            = _config.getPostfix() ;
        _prefix             = _config.getPrefix() ;
        _indent             = _config.getIndent();
        _legacy             = _config.getIsLegacy();
    }

    /**
     *  This method is a wrapper for the _asStr.append() call.
     *  It thereby greatly reduces boilerplate code which
     *  makes this class substantially more readable,
     *  however one has to keep in mind that "_$" string is
     *  a chained builder method.
     *
     * @param toBeAppended The String which ought to be appended to this builder.
     * @return This very instance in order to enable method-chaining.
     */
    private NdaAsString _$( String toBeAppended ) {
        _asStr.append( toBeAppended ); return this;
    }

    /**
     *  This method is a wrapper for the _asStr.append() call.
     *  It thereby greatly reduces boilerplate code which
     *  makes this class substantially more readable,
     *  however one has to keep in mind that "_$" string is
     *  a chained builder method.
     *
     * @param toBeAppended The int which ought to be appended to this builder.
     * @return This very instance in order to enable method-chaining.
     */
    private NdaAsString _$(int toBeAppended ) { _asStr.append( toBeAppended ); return this; }

    /**
     *  This method takes the data of a tensor and converts it into
     *  a lambda responsible for creating formatted Strings of data entries by
     *  taking the int index of a targeted entry and returning
     *  a properly formatted String representation of said entry. <br>
     *  <br>
     *
     * @param data The data which should be used as a basis for creating an entry stringification lambda.
     * @return A lambda which can convert an array entry targeted by its index to a properly formatted String.
     */
    private ValStringifier _createValStringifierAndFormatter( Object data )
    {
        final ValStringifier function = _createBasicStringifierFor( data, _isCompact );
        int cellSize = _cellSize;
        final ValStringifier postProcessing;
        if ( cellSize >= 3 ) postProcessing = i -> {
            String s = function.stringify( i );
            int margin = cellSize - s.length();
            int right = ( margin % 2 == 0 ) ? margin / 2 : ( margin-1 ) / 2;
            if ( margin > 0 ) s = Util.pad( margin - right, Util.pad( s, right ) );
            return s;
        };
        else postProcessing = function;

        final ValStringifier finalProcessing;
        if ( _isCellBound ) finalProcessing = i -> {
            String s = postProcessing.stringify( i );
            int margin =  s.length() - cellSize;
            if ( margin > 0 )
                s = s.substring(0, Math.max(0, cellSize - 2)) + "..";

            return s;
        };
        else finalProcessing = postProcessing;

        return finalProcessing;
    }

    private Function<String, String> _createStringItemFilter() {
        if ( _haveSlimNumbers )
            return vStr -> {
                        if ( vStr.endsWith("E0")   ) vStr = vStr.substring( 0, vStr.length() - 2 );
                        if ( vStr.endsWith(".0")   ) vStr = vStr.substring( 0, vStr.length() - 2 );
                        return vStr.startsWith("0.") ? vStr.substring( 1 ) : vStr;
                    };
        if ( Number.class.isAssignableFrom(_tensor.itemType()) ) {
            if (
                _tensor.itemType() == Integer.class ||
                _tensor.itemType() == Byte.class    ||
                _tensor.itemType() == Long.class    ||
                _tensor.itemType() == Short.class
            )
                return  vStr -> {
                    if ( vStr.endsWith("E0") ) vStr = vStr.substring( 0, vStr.length() - 2 );
                    return vStr.endsWith(".0") ? vStr.substring( 0, vStr.length() - 2 ) : vStr;
                };
        }
        return vStr -> vStr.endsWith("E0") ? vStr.substring( 0, vStr.length() - 2 ) : vStr;
    }

    /**
     *  @param data The data array which may be an array of primitives or reference objects.
     *  @param isCompact A flag determining if the array items should be formatted compactly or not.
     *  @return An {@link ValStringifier} which can return {@link String} representations of array items targeted via their index.
     */
    private ValStringifier _createBasicStringifierFor( Object data, boolean isCompact )
    {
        if ( data == null )
            throw new IllegalArgumentException("Tensor has no value data!");

        Function<String, String> slim = _createStringItemFilter();

        if ( data instanceof double[] )
            return i -> ( isCompact )
                    ? slim.apply( formatFP( ( (double[]) data )[ i ]) )
                    : slim.apply( String.valueOf( ( (double[] ) data )[ i ] ) );
        else if ( data instanceof float[] )
            return i -> ( isCompact )
                    ? slim.apply( formatFP( ( (float[]) data )[ i ] ) )
                    : slim.apply( String.valueOf( ( (float[]) data )[ i ] ) );
        else if ( data instanceof short[] )
            return i -> ( isCompact )
                    ? slim.apply( formatFP( ( (short[]) data )[ i ] ))
                    : slim.apply( String.valueOf( ( (short[]) data )[ i ] ));
        else if ( data instanceof int[] )
            return i -> ( isCompact )
                    ? slim.apply( formatFP( ( (int[]) data )[ i ] ))
                    : slim.apply( String.valueOf( ( (int[]) data )[ i ] ));
        else if ( data instanceof byte[] )
            return i -> ( isCompact )
                    ? slim.apply( formatFP( ( (byte[]) data )[ i ] ))
                    : slim.apply( String.valueOf( ( (byte[]) data )[ i ] ));
        else if ( data instanceof long[] )
            return i -> ( isCompact )
                    ? slim.apply( formatFP( ( (long[]) data )[ i ] ))
                    : slim.apply( String.valueOf( ( (long[]) data )[ i ] ));
        else if ( data instanceof boolean[] )
            return i -> String.valueOf( ( (boolean[]) data )[ i ] );
        else if ( data instanceof char[] )
            return i -> String.valueOf( ( (char[]) data )[ i ] );
        else
            return i -> String.valueOf( ( (Object[]) data )[ i ] );
    }

    public String toString() {
        if ( _tensor.isDeleted() ) return "deleted";
        else if ( _tensor.isEmpty() ) return "empty";
        else if ( _tensor.isUndefined() ) return "undefined";
        _asStr = new StringBuilder();
        String base      = ( !_isMultiline ? ""   : _breakAndIndent() + "    " );
        String delimiter = ( !_isMultiline ? ""   : "    "                     );
        String half      = ( !_isMultiline ? ""   : "  "                       );
        if ( _hasShape ) _strShape();
        if ( !_hasValue ) return _asStr.toString();
        _$( ":" );
        if ( _isMultiline ) _recursiveFormatting( new int[ _tensor.rank() ], -1 );
        else {
            _$( _legacy ? "(" : "[" );
            _stringifyAllValues();
            _$( _legacy ? ")" : "]" );
        }

        if ( _hasGradient && ( _tensor.rqsGradient() || _tensor.hasGradient() ) ) {
            _$( ":g" );
            Tsr<?> gradient = _tensor.getGradient();
            if ( gradient != null )
                _$(
                    gradient.toString(
                        t -> t.with( _config )
                                .setPrefix("").setPostfix("")
                                .setHasShape( false )
                                .setIsScientific( true )
                    )
                );
            else
                _$( ( ( _legacy ) ? ":(null)" : ":[null]" ) );
        }
        if ( _hasRecursiveGraph && _tensor.has( GraphNode.class ) && _tensor.get( GraphNode.class ).size() > 0 ) {
            GraphNode<?> node = _tensor.get( GraphNode.class );
            _$( "; " );
            node.forEachDerivative( ( t, agent ) -> {
                if ( !agent.partialDerivative().isPresent() ) _$( "->d(null), " );
                else _$(
                        base + "=>d|[ " +
                        base + delimiter + agent.partialDerivative().get().toString( _config.clone().setPrefix("").setPostfix("") ) + " " +
                        base + half + "]|:t{ " +
                        base + delimiter + (
                            ( t.getPayload() != null ) ? t.getPayload().toString( _config.clone().setPrefix("").setPostfix("") ) : t.toString()
                        ) +
                        " " + base + half + "}, "
                    );
            });
        }
        if ( _hasDerivatives && _tensor.has( GraphNode.class ) && _tensor.get( GraphNode.class ).size() > 0 ) {
            GraphNode<?> node = _tensor.get( GraphNode.class );
            if ( node.getMode() != 0 ) {
                _$( "; " );
                node.forEachDerivative( ( t, action ) -> {
                    if ( !action.partialDerivative().isPresent() ) _$( "->d(" )._$( action.toString() )._$( "), " );
                    else _$( "->d" )._$( action.partialDerivative().get().toString( _config.clone().setPrefix("").setPostfix("") ) )._$( ", " );
                });
            }
        }
        return _postProcessed( _asStr.toString() );
    }

    private String _postProcessed( String asString )
    {
        asString =  asString.trim(); // We need to get rid of line breaks

        if ( asString.endsWith(",") )
            asString = asString.substring(0, asString.length()-1);

        asString = _prefix + asString + _postfix;

        return asString;
    }

    private void _stringifyAllValues()
    {
        int max = _rowLimit;
        Object data = ( _tensor.isOutsourced() ? _tensor.getRawData() : _tensor.getMut().getData().getRef() );
        ValStringifier getter = _createValStringifierAndFormatter( data );
        int size = _tensor.size();
        int trim = ( size - max );
        size = ( trim > 0 ? max : size );
        for ( int i = 0; i < size; i++ ) {
            String vStr = getter.stringify( ( _tensor.isVirtual() ) ? 0 : _tensor.indexOfIndex( i ) );
            _$( vStr );
            if ( i < size - 1 ) _$( ", " );
            else if ( trim > 0 ) _$( ", ... + " )._$( trim )._$( " more" );
        }
    }

    /**
     *  This method builds a single row of stringified and formatted tensor entries.
     *  It will build this row based on incrementing the last tensor dimension. <br>
     *  <br>
     * @param trimStart The row index where the trimming should start (no entries).
     * @param trimEnd The row index where the trimming should end.
     * @param trimSize The size of the row chunk which ought to be skipped.
     * @param indices The current index array defining the current position inside the tensor.
     * @param stringifier A lambda responsible for stringifying a tensor entry ba passing the current index array.
     * @param delimiter The String which ought to separate stringified entries.
     */
    private void _buildRow(
            int trimStart, int trimEnd, int trimSize, int[] indices, NDValStringifier stringifier, String delimiter
    ) {
        for ( int i = 0; i < _shape[ _shape.length - 1 ]; i++ ) {
            if ( i <= trimStart || i > trimEnd ) {
                _$( stringifier.stringify( indices ) );
                if ( i < _shape[ _shape.length - 1 ] - 1 ) _$( delimiter );
            }
            else if ( i == trimStart + 1 ) _$( ".." )._$( trimSize )._$( " more.., " );

            NDConfiguration.Utility.increment( indices, _shape );
        }
    }


    /**
     *  This method builds a properly indented and formatted tensor representation.
     *  The depth of the recursion is also tracked by the current dimension which
     *  will be incrementally passed down the recursion. <br>
     *  Besides this dimension there is also the current index array
     *  which will be incremented when tensor elements are being stringified... <br>
     *  <br>
     *
     * @param indices The current index array containing the current index for all dimensions.
     * @param dim The current dimension which is also the "depth" of the recursion.
     */
    private void _recursiveFormatting( int[] indices, int dim )
    {
        dim = Math.max( dim, 0 );
        int trimSize = ( _shape[ dim ] - _rowLimit );
        trimSize = Math.max( trimSize, 0 );
        int trimStart = ( _shape[ dim ] / 2 - trimSize / 2 );
        int trimEnd = ( _shape[ dim ] / 2 + trimSize / 2 );
        trimEnd += ( trimSize % 2 );
        if ( !(_shape[dim] <= _rowLimit || (_shape[dim] - (trimEnd - trimStart)) == _rowLimit) )
            throw new IllegalStateException("Failed to print tensor!");
        NDFrame<?> alias = _tensor.get( NDFrame.class );
        if ( dim == indices.length - 1 ) {
            if (
                alias != null &&
                indices[ indices.length - 1 ] == 0 &&
                indices[ Math.max( indices.length - 2, 0 ) ] == 0
            ) {
                List<Object> aliases = alias.atAxis( indices.length - 1 ).getAllAliases();
                if ( aliases != null ) {
                    _$( Util.indent( dim ) );
                    _$( _legacy ? "[ " : "( " ); // The following assert has prevented many String miscarriages!
                    int missing = _shape[ indices.length - 1 ] - aliases.size();
                    if ( missing > 0 ) { // This is a basic requirement for the label size...
                        aliases = new ArrayList<>(aliases);
                        for ( int i = 0; i < missing; i++ ) aliases.add("");
                    }
                    ValStringifier getter = _createValStringifierAndFormatter( aliases.toArray() );
                    _buildRow(
                        trimStart, trimEnd, trimSize,
                        new int[ indices.length ],
                        iarr -> getter.stringify( iarr[ iarr.length -1 ] ),
                        _legacy ? "][" : ")("
                    );
                    _$( _legacy ? " ]" : " )" );
                    if ( alias.getLabel() != null )
                        _$( (_legacy) ? ":[ " : ":( " )._$( alias.getLabel() )._$( (_legacy) ? " ]" : " )" );
                    _$( _breakAndIndent() );
                }
            }
            _$( Util.indent( dim ) );
            _$( _legacy ? "( " : "[ " );
            ValStringifier getter = _createValStringifierAndFormatter( _tensor.getRawData() );
            NDValStringifier fun = _tensor.isVirtual()
                                        ? iarr -> getter.stringify( 0 )
                                        : iarr -> getter.stringify( _tensor.indexOfIndices( iarr ) );

            _buildRow( trimStart, trimEnd, trimSize, indices, fun, ", " );
            _$( _legacy ? " )" : " ]" );

            if ( alias != null && alias.hasLabelsForAxis( dim - 1 ) )
                _$( ":" )._buildSingleLabel( alias, dim, indices );
        } else {
            _$( Util.indent( dim ) );
            if ( dim > 0 && alias != null && alias.hasLabelsForAxis(dim-1) )
                _buildSingleLabel( alias, dim, indices )._$(":");
            _$( ( _legacy ? "(" : "[" ) + _breakAndIndent() );
            int i = 0;
            do {
                if ( i < trimStart || i >= trimEnd )
                    _recursiveFormatting( indices, dim + 1 );
                else if ( i == trimStart )
                    _$( Util.indent( dim + 1 ) )._$( "... " )._$( trimSize )._$( " more ..." + _breakAndIndent() );
                else
                    indices[ dim ] = trimEnd + 1; // Jumping over trimmed entries!
                i++;
            }
            while ( indices[ dim ] != 0 );
            _$( Util.indent( dim ) )._$( _legacy ? ")" : "]" );
        }
        int i = dim - 1;
        if (i >= 0 && indices[i] != 0) _$( "," );
        if ( i >= 0 ) _$( _breakAndIndent() );
    }

    private NdaAsString _buildSingleLabel(NDFrame<?> alias, int dim, int[] indices ) {
        int pos = dim - 1;
        List<Object> key = alias.atAxis( pos ).getAllAliases();
        if ( pos >= 0 && key != null ) {
            _$( _legacy ? "[ " : "( " );
            int i = ( dim == indices.length - 1 )
                    ? ( _shape[ pos ] + indices[ pos ] - 1 ) % _shape[ pos ]
                    : indices[ pos ];
            _$( i >= key.size() ? "" : key.get( i ).toString() );
            _$( _legacy ? " ]" : " )" );
        }
        return this;
    }

    /**
     *  This method builds a String representation of the
     *  shape of the targeted tensor.
     */
    private void _strShape()
    {
        boolean legacy = _legacy;
        _$( legacy ? "[" : "(" );
        for ( int i = 0; i < _shape.length; i++ ) {
            _$( _shape[ i ] );
            if ( i < _shape.length - 1 ) _$( "x" );
        }
        _$( legacy ? "]" : ")" );
    }

    private String _breakAndIndent() {
        return "\n" + _indent;
    }

    
    private static String formatFP( double v )
    {
        DecimalFormatSymbols formatSymbols = new DecimalFormatSymbols( Locale.UK );
        DecimalFormat formatter = new DecimalFormat("##0.0##E0", formatSymbols);
        String vStr = String.valueOf( v );
        final int offset = 0;
        if ( vStr.length() > ( 7 - offset ) ) {
            if ( vStr.startsWith("0.") )
                vStr = vStr.substring( 0, 7 - offset ) + "E0";
            else if ( vStr.startsWith( "-0." ) )
                vStr = vStr.substring( 0, 8 - offset ) + "E0";
            else {
                vStr = formatter.format( v );
                vStr = !vStr.contains(".0E0") ? vStr : vStr.replace(".0E0",".0");
                vStr =  vStr.contains(".")    ? vStr : vStr.replace("E0",".0");
            }
        }
        if ( vStr.endsWith("E0") ) vStr = vStr.substring(0, vStr.length()-2);
        return vStr.replace("E", "e");
    }

    /**
     *  A builder interface providing multiple different options for building
     *  a {@link NdaAsString} instance in a fluent way.
     */
    public interface Builder {
        /**
         * @param configMap The configuration map used as basis for turning the wrapped {@link Tsr} to a {@link String}.
         * @return A new {@link NdaAsString} based on the provided configuration.
         */
        NdaAsString withConfig( NDPrintSettings configMap );

        /**
         * @param config The configuration used as basis for turning the wrapped {@link Tsr} to a {@link String}.
         * @return A new {@link NdaAsString} based on the provided configuration.
         */
        NdaAsString withConfig( String config );

        /**
         * @return A new {@link NdaAsString} based on the default configuration.
         */
        NdaAsString byDefaults();
    }

    /**
     *  A simple functional interface for turning something at a particular
     *  index into a {@link String}.
     */
    @FunctionalInterface
    private interface ValStringifier
    {
        String stringify( int i );
    }

    /**
     *  A simple functional interface for turning something at a particular
     *  multidimensional index into a {@link String}.
     */
    @FunctionalInterface
    private interface NDValStringifier
    {
        String stringify( int[] idx );
    }

    /**
     *  This class is a simple utility class which contains
     *  a collection of static and stateless methods containing
     *  useful functionalities for tensor stringification.
     */
    public static class Util
    {
        
        public static String indent( int n ) { return String.join("", Collections.nCopies( n, "   " )); }

        
        public static String pad( int left, String s ) {
            return String.join("", Collections.nCopies( left, " " )) + s;
        }

        
        public static String pad( String s, int right ) {
            return s + String.join("", Collections.nCopies( right, " " ));
        }

    }


}
