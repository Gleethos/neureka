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

package neureka.utility;

import neureka.Neureka;
import neureka.Tsr;
import neureka.autograd.GraphNode;
import neureka.framing.NDFrame;
import neureka.ndim.config.NDConfiguration;
import org.jetbrains.annotations.Contract;

import java.text.DecimalFormat;
import java.text.DecimalFormatSymbols;
import java.util.*;
import java.util.function.Function;

/**
 *  This class is in essence a simple wrapper class for a tensor and a StringBuilder
 *  Methods in this class use the builder in order to construct a String representation
 *  for said tensor.
 *  These method adjust this String building based on a set of certain
 *  configurations which are represented by the "Should" enum.
 *
 */
public final class TsrAsString
{
    private int _padding = 6;
    private int _shortage = 50;
    private boolean _hasGradient = true;
    private boolean _isCompact = true;
    private boolean _isFormatted = true;
    private boolean _hasValue = true;
    private boolean _hasShape = true;
    private boolean _hasRecursiveGraph = false;
    private boolean _hasDerivatives = false;
    private boolean _isCellBound = false;

    private int[] _shape;
    private Tsr<?> _tensor;
    private StringBuilder _asStr;
    private final boolean _legacy = Neureka.get().settings().view().isUsingLegacyView();

    private Map<Should, Object> _config;

    public enum Should {
        BE_FORMATTED,
        HAVE_PADDING_OF,
        BE_COMPACT,
        HAVE_GRADIENT,
        BE_SHORTENED_BY,
        HAVE_VALUE,
        HAVE_SHAPE,
        HAVE_DERIVATIVES,
        HAVE_RECURSIVE_GRAPH,
        BE_CELL_BOUND
    }

    private interface ValStringifier {
        String stringify( int i );
    }

    public TsrAsString( Tsr<?> tensor, Map< Should, Object > settings )
    {
        Map< Should, Object > copy = Util.configFromCode( "" );
        for ( Should s : Should.values() )
            if ( settings.containsKey( s ) ) copy.put( s, settings.get( s ) );
        _construct( tensor, copy );
    }

    public TsrAsString( Tsr<?> tensor, String modes )
    {
        _construct(
                tensor,
                Util.configFromCode( modes )
        );
    }

    public TsrAsString( Tsr<?> tensor )
    {
        _construct(
                tensor,
                Util.configFromCode( null )
        );
    }

    private void _construct( Tsr<?> tensor, Map< Should, Object > settings )
    {
        if ( tensor.getNDConf() != null ) _shape = tensor.getNDConf().shape();
        _config = settings;
        _tensor = tensor;
        // TODO: Add some asserts!
        if ( settings.containsKey( Should.BE_COMPACT ) )
            _isCompact = (boolean) settings.get( Should.BE_COMPACT );

        if ( settings.containsKey( Should.BE_SHORTENED_BY ) )
            _shortage = (int) settings.get( Should.BE_SHORTENED_BY );

        if ( settings.containsKey( Should.HAVE_PADDING_OF ) )
            _padding = (int) settings.get( Should.HAVE_PADDING_OF );

        if ( settings.containsKey( Should.HAVE_GRADIENT ) )
            _hasGradient = (boolean) settings.get( Should.HAVE_GRADIENT );

        if ( settings.containsKey( Should.BE_FORMATTED ) )
            _isFormatted = (boolean) settings.get( Should.BE_FORMATTED );

        if ( settings.containsKey( Should.HAVE_VALUE ) )
            _hasValue = (boolean) settings.get( Should.HAVE_VALUE );

        if ( settings.containsKey( Should.HAVE_SHAPE ) )
            _hasShape = (boolean) settings.get( Should.HAVE_SHAPE );

        if ( settings.containsKey( Should.HAVE_RECURSIVE_GRAPH) )
            _hasRecursiveGraph = (boolean) settings.get( Should.HAVE_RECURSIVE_GRAPH);

        if ( settings.containsKey( Should.HAVE_DERIVATIVES) )
            _hasDerivatives = (boolean) settings.get( Should.HAVE_DERIVATIVES);

        if ( settings.containsKey( Should.BE_CELL_BOUND ) )
            _isCellBound = (boolean) settings.get( Should.BE_CELL_BOUND );
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
    private TsrAsString _$( String toBeAppended ) { _asStr.append( toBeAppended ); return this; }

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
    private TsrAsString _$( int toBeAppended ) { _asStr.append( toBeAppended ); return this; }

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
        int pad = _typeAdjustedPadding();
        final ValStringifier postProcessing;
        if ( pad >= 3 ) postProcessing = i -> {
            String s = function.stringify( i );
            int margin = pad - s.length();
            int right = ( margin % 2 == 0 ) ? margin / 2 : ( margin-1 ) / 2;
            if ( margin > 0 ) s = Util.pad( margin - right, Util.pad( s, right ) );
            return s;
        };
        else postProcessing = function;

        final ValStringifier finalProcessing;
        if ( _isCellBound ) finalProcessing = i -> {
            String s = postProcessing.stringify( i );
            int margin =  s.length() - pad;
            if ( margin > 0 ) s = s.substring( 0, pad - 2 ) + "..";
            return s;
        };
        else finalProcessing = postProcessing;

        return finalProcessing;
    }

    /**
     *  @param data The data array which may be an array of primitives or reference objects.
     *  @param isCompact A flag determining if the array items should be formatted compactly or not.
     *  @return An {@link ValStringifier} which can return {@link String} representations of array items targeted via their index.
     */
    private ValStringifier _createBasicStringifierFor( Object data, boolean isCompact )
    {
        if ( data instanceof double[] )
            return i -> ( isCompact )
                    ? Util.formatFP( ( (double[]) data )[ i ])
                    : String.valueOf( ( (double[] ) data )[ i ] );
        else if ( data instanceof float[] )
            return i -> ( isCompact )
                    ? Util.formatFP( ( (float[]) data )[ i ] )
                    : String.valueOf( ( (float[]) data )[ i ] );
        else if ( data instanceof short[] )
            return i -> ( isCompact )
                    ? Util.formatFP( ( (short[]) data )[ i ] )
                    : String.valueOf( ( (short[]) data )[ i ] );
        else if ( data instanceof int[] )
            return i -> ( isCompact )
                    ? Util.formatFP( ( (int[]) data )[ i ] )
                    : String.valueOf( ( (int[]) data )[ i ] );
        else if ( data instanceof byte[] )
            return i -> ( isCompact )
                    ? Util.formatFP( ( (byte[]) data )[ i ] )
                    : String.valueOf( ( (byte[]) data )[ i ] );
        else if ( data instanceof long[] )
            return i -> ( isCompact )
                    ? Util.formatFP( ( (long[]) data )[ i ] )
                    : String.valueOf( ( (long[]) data )[ i ] );
        else if ( data == null )
            return i -> ( isCompact )
                    ? Util.formatFP( _tensor.value64( i ) )
                    : String.valueOf( _tensor.value64( i ) );
        else
            return i -> String.valueOf( ( (Object[]) data )[ i ] );
    }

    /**
     * @return A potentially modified version of the configured padding to better suite certain types.
     */
    private int _typeAdjustedPadding() {
        if ( _tensor.getDataType().getTypeClass() == String.class )
            return  (int)(_padding * 2.5);
        else
            return  _padding;
    }

    public String toString() { return toString(""); }

    public String toString( String depth )
    {
        if ( _tensor.isEmpty() ) return "empty";
        else if ( _tensor.isUndefined() ) return "undefined";
        _asStr = new StringBuilder();
        String base      = ( depth == null ? ""   : "\n" + depth      );
        String delimiter = ( depth == null ? ""   : "    "            );
        String half      = ( depth == null ? ""   : "  "              );
        String deeper    = ( depth == null ? null : depth + delimiter );
        if ( _hasShape ) _strShape();
        if ( !_hasValue ) return _asStr.toString();
        _$( ":" );
        if ( _isFormatted ) _recursiveFormatting( new int[ _tensor.rank() ], -1 );
        else {
            _$( (_legacy) ? "(" : "[" );
            _stringifyAllValues();
            _$( (_legacy) ? ")" : "]" );
        }

        if ( _hasGradient && ( _tensor.rqsGradient() || _tensor.hasGradient() ) ) {
            _$( ":g" );
            Tsr<?> gradient = _tensor.getGradient();
            if ( gradient != null )
                _$( gradient.toString( "cv" ) );
            else
                _$( ( ( _legacy ) ? ":(null)" : ":[null]" ) );
        }
        if ( _hasRecursiveGraph && _tensor.has( GraphNode.class ) && _tensor.get( GraphNode.class ).size() > 0 ) {
            GraphNode<?> node = _tensor.get( GraphNode.class );
            _$( "; " );
            node.forEachDerivative( ( t, agent ) -> {
                if ( agent.derivative() == null ) _$( "->d(null), " );
                else {
                    _$(
                                    base + "=>d|[ " +
                                    base + delimiter + agent.derivative().toString( _config, deeper ) + " " +
                                    base + half + "]|:t{ " +
                                    base + delimiter + (
                                    ( t.getPayload() != null ) ? t.getPayload().toString( _config, deeper ) : t.toString("")
                            ) + " " + base + half + "}, "
                    );
                }
            });
        }
        if ( _hasDerivatives && _tensor.has( GraphNode.class ) && _tensor.get( GraphNode.class ).size() > 0 ) {
            GraphNode<?> node = _tensor.get( GraphNode.class );
            if ( node.getMode() != 0 ) {
                _$( "; " );
                node.forEachDerivative( ( t, agent ) -> {
                    if ( agent.derivative() == null ) _$( "->d(" )._$( agent.toString() )._$( "), " );
                    else _$( "->d" )._$( agent.derivative().toString( _config, deeper ) )._$( ", " );
                });
            }
        }
        return _asStr.toString();
    }

    private void _stringifyAllValues()
    {
        int max = _shortage;
        ValStringifier getter = _createValStringifierAndFormatter( _tensor.getData() );
        int size = _tensor.size();
        int trim = ( size - max );
        size = ( trim > 0 ) ? max : size;
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
            int trimStart, int trimEnd, int trimSize, int[] indices, Function<int[],String> stringifier, String delimiter
    ) {
        for ( int i = 0; i < _shape[ _shape.length - 1 ]; i++ ) {
            if ( i < trimStart || i >= trimEnd ) {
                _$( stringifier.apply( indices ) );
                if ( i < _shape[ _shape.length - 1 ] - 1 ) _$( delimiter );
            }
            else if ( i == trimStart ) _$( "... " )._$( trimSize )._$( " more ..., " );

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
        int max = ( _shortage * 32 / 50 );
        dim = Math.max( dim, 0 );
        int trimSize = ( _shape[ dim ] - max );
        trimSize = Math.max( trimSize, 0 );
        int trimStart = ( _shape[ dim ] / 2 - trimSize / 2 );
        int trimEnd = ( _shape[ dim ] / 2 + trimSize / 2 );
        assert trimEnd - trimStart == trimSize;
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
                    _$( (_legacy) ? "[ " : "( " ); // The following assert has prevented many String miscarriages!
                    assert aliases.size() - _shape[ indices.length - 1 ] == 0; // This is a basic requirement for the label size...
                    ValStringifier getter = _createValStringifierAndFormatter( aliases.toArray() );
                    _buildRow(
                            trimStart, trimEnd, trimSize,
                            new int[ indices.length ],
                            iarr -> getter.stringify( iarr[ iarr.length -1 ] ),
                            (_legacy) ? "][" : ")("
                    );
                    _$( (_legacy) ? " ]" : " )" );
                    if ( alias.getTensorName() != null )
                        _$( (_legacy) ? ":[ " : ":( " )._$( alias.getTensorName() )._$( (_legacy) ? " ]" : " )" );
                    _$( "\n" );
                }
            }
            _$( Util.indent( dim ) );
            _$( _legacy ? "( " : "[ " );
            ValStringifier getter = _createValStringifierAndFormatter( _tensor.getData() );
            Function<int[], String> fun = ( _tensor.isVirtual() )
                    ? iarr -> getter.stringify( 0 )
                    : iarr -> getter.stringify( _tensor.indexOfIndices( iarr ) );

            _buildRow( trimStart, trimEnd, trimSize, indices, fun, ", " );
            _$( _legacy ? " )" : " ]" );

            if ( alias != null ) _$( ":" )._buildSingleLabel( alias, dim, indices );
        } else {
            _$( Util.indent( dim ) );
            if ( dim > 0 && alias != null )
                _buildSingleLabel( alias, dim, indices )._$(":");
            _$( _legacy ? "(\n" : "[\n" );
            int i = 0;
            do {
                if ( i < trimStart || i >= trimEnd )
                    _recursiveFormatting( indices, dim + 1 );
                else if ( i == trimStart )
                    _$( Util.indent( dim + 1 ) )._$( "... " )._$( trimSize )._$( " more ...\n" );
                else
                    indices[ dim ] = trimEnd + 1; // Jumping over trimmed entries!
                i++;
            }
            while ( indices[ dim ] != 0 );
            _$( Util.indent( dim ) )._$( (_legacy) ? ")" : "]" );
        }
        int i = dim - 1;
        if ( i >= 0 && i < indices.length && indices[ i ] != 0 ) _$( "," );
        _$( "\n" );
    }

    private TsrAsString _buildSingleLabel( NDFrame<?> alias, int dim, int[] indices ) {
        int pos = dim - 1;
        List<Object> key = alias.atAxis( pos ).getAllAliases();
        if ( pos >= 0 && key != null ) {
            _$( _legacy ? "[ " : "( " );
            int i = ( dim == indices.length - 1 )
                    ? ( _shape[ pos ] + indices[ pos ] - 1 ) % _shape[ pos ]
                    : indices[ pos ];
            _$( key.get( i ).toString() );
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
        boolean legacy = Neureka.get().settings().view().isUsingLegacyView();
        _$( legacy ? "[" : "(" );
        for ( int i = 0; i < _shape.length; i++ ) {
            _$( _shape[ i ] );
            if ( i < _shape.length - 1 ) _$( "x" );
        }
        _$( legacy ? "]" : ")" );
    }

    /**
     *  This class is a simple utility class which contains
     *  a collection of static and stateless methods containing
     *  useful functionalities for tensor stringification.
     */
    public static class Util
    {
        @Contract( pure = true )
        public static String indent( int n ) { return String.join("", Collections.nCopies( n, "   " )); }

        @Contract( pure = true )
        public static String pad( int left, String s ) {
            return String.join("", Collections.nCopies( left, " " )) + s;
        }

        @Contract( pure = true )
        public static String pad( String s, int right ) {
            return s + String.join("", Collections.nCopies( right, " " ));
        }

        @Contract( pure = true )
        public static String formatFP( double v )
        {
            DecimalFormatSymbols formatSymbols = new DecimalFormatSymbols( Locale.US );
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
            return vStr;
        }

        @Contract( pure = true )
        public static Map<Should, Object> configFromCode( String modes )
        {
            if ( modes == null )
                return Neureka.get().settings().view().getAsString();

            Map< Should, Object > conf = new HashMap<>();
            conf.put( Should.BE_SHORTENED_BY,      modes.contains( "s" ) ? 3 : 50                             );
            conf.put( Should.BE_COMPACT,           modes.contains( "c" )                                      );
            conf.put( Should.BE_FORMATTED,         modes.contains( "f" )                                      );
            conf.put( Should.HAVE_GRADIENT,        modes.contains( "g" )                                      );
            conf.put( Should.HAVE_PADDING_OF,      modes.contains( "p" ) ? 6 : modes.contains( "f" ) ? 2 : 1  );
            conf.put( Should.HAVE_VALUE,          !(modes.contains( "shp" ) || modes.contains("shape"))       );
            conf.put( Should.HAVE_RECURSIVE_GRAPH, modes.contains( "r" )                                      );
            conf.put( Should.HAVE_DERIVATIVES,     modes.contains( "d" )                                      );
            conf.put( Should.HAVE_SHAPE,           !modes.contains( "v" )                                     );
            conf.put( Should.BE_CELL_BOUND,        modes.contains( "b" )                                      );
            return conf;
        }
    }


}
