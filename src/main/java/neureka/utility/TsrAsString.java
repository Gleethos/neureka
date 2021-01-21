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

import lombok.Getter;
import lombok.experimental.Accessors;
import neureka.Neureka;
import neureka.Tsr;
import neureka.autograd.GraphNode;
import neureka.framing.IndexAlias;
import neureka.ndim.AbstractNDArray;
import neureka.ndim.config.NDConfiguration;
import org.jetbrains.annotations.Contract;

import java.text.DecimalFormat;
import java.text.DecimalFormatSymbols;
import java.util.*;
import java.util.function.Function;
import java.util.function.IntFunction;
import java.util.stream.Collectors;

/**
 *  This class is in essence a simple wrapper class for a tensor and a StringBuilder
 *  Methods in this class use the builder in order to construct a String representation
 *  for said tensor.
 *  These method adjust this String building based on a set of certain
 *  configurations which are represented by the "Should" enum.
 *
 */
@Accessors( prefix = {"_"} )
public class TsrAsString
{
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

    @Getter private int _padding = 6;
    @Getter private int _shortage = 50;
    @Getter private boolean _hasGradient = true;
    @Getter private boolean _isCompact = true;
    @Getter private boolean _isFormatted = true;
    @Getter private boolean _hasValue = true;
    @Getter private boolean _hasShape = true;
    @Getter private boolean _hasRecursiveGraph = false;
    @Getter private boolean _hasDerivatives = false;
    @Getter private boolean _isCellBound = false;

    private int[] _shape;
    private Tsr<?> _tensor;
    private StringBuilder _asStr;
    private final boolean _legacy = Neureka.instance().settings().view().isUsingLegacyView();

    private Map<Should, Object> _config;

    public TsrAsString( Tsr<?> tensor, Map< Should, Object > settings ) {
        Map< Should, Object > copy = _defaults( "" );
        for ( Should s : Should.values() )
            if ( settings.containsKey( s ) ) copy.put( s, settings.get( s ) );
        _construct( tensor, copy );
    }

    public TsrAsString( Tsr<?> tensor, String modes ) {
        _construct(
                tensor,
                _defaults( modes )
        );
    }

    public TsrAsString( Tsr<?> tensor ) {
        _construct(
                tensor,
                _defaults( null )
        );
    }

    private void _construct( Tsr tensor, Map< Should, Object > settings )
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

    private Map< Should, Object > _defaults( String modes ) {
        if ( modes == null ) {
            return Neureka.instance().settings().view().getAsString();
        }
        return configFromCode( modes );
    }

    public static Map<Should, Object> configFromCode( String code ) {
        Map< Should, Object > copy = new HashMap<>();
        copy.put( Should.BE_SHORTENED_BY,      (code.contains( "s") ) ? 3 : 50                      );
        copy.put( Should.BE_COMPACT,           code.contains( "c" )                                 );
        copy.put( Should.BE_FORMATTED,         code.contains( "f" )                                 );
        copy.put( Should.HAVE_GRADIENT,        code.contains( "g" )                                 );
        copy.put( Should.HAVE_PADDING_OF,     (code.contains( "p" )) ? 6 : code.contains( "f" )?2:1 );
        copy.put( Should.HAVE_VALUE,          !(code.contains( "shp" ) || code.contains("shape"))   );
        copy.put( Should.HAVE_RECURSIVE_GRAPH, code.contains( "r" )                                 );
        copy.put( Should.HAVE_DERIVATIVES,     code.contains( "d" )                                 );
        copy.put( Should.HAVE_SHAPE,           !code.contains( "v" )                                );
        copy.put( Should.BE_CELL_BOUND,        code.contains("b")                                   );
        return copy;
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
    private TsrAsString _$( String toBeAppended ) {
        _asStr.append( toBeAppended );
        return this;
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
    private TsrAsString _$( int toBeAppended ) {
        _asStr.append( toBeAppended );
        return this;
    }

    private IntFunction<String> _createValStringifier( Object v )
    {
        boolean compact = _isCompact;
        int pad = ( _tensor.getDataType().getTypeClass() == String.class )
                ? (int)(_padding * 2.5)
                : _padding;
        final IntFunction<String> function;
        if ( v instanceof double[] )
            function = i -> ( compact )
                    ? Util.formatFP( ( (double[]) v )[ i ])
                    : String.valueOf( ( (double[] ) v )[ i ] );
        else if ( v instanceof float[] )
            function = i -> ( compact )
                    ? Util.formatFP( ( (float[]) v )[ i ] )
                    : String.valueOf( ( (float[]) v )[ i ] );
        else if ( v instanceof short[] )
            function = i -> ( compact )
                    ? Util.formatFP( ( (short[]) v )[ i ] )
                    : String.valueOf( ( (short[]) v )[ i ] );
        else if ( v instanceof int[] )
            function = i -> ( compact )
                    ? Util.formatFP( ( (int[]) v )[ i ] )
                    : String.valueOf( ( (int[]) v )[ i ] );
        else if ( v == null )
            function = i -> ( compact )
                    ? Util.formatFP( _tensor.value64( i ) )
                    : String.valueOf( _tensor.value64( i ) );
        else
            function = i -> String.valueOf( ( (Object[]) v )[ i ] );

        final IntFunction<String> postProcessing;
        if ( pad >= 3 ) postProcessing = i -> {
            String s = function.apply( i );
            int margin = pad - s.length();
            int right = ( margin % 2 == 0 ) ? margin / 2 : ( margin-1 ) / 2;
            if ( margin > 0 ) s = Util.pad( margin - right, Util.pad( s, right ) );
            return s;
        };
        else postProcessing = function;

        final IntFunction<String> finalProcessing;
        if ( _isCellBound ) finalProcessing = i -> {
            String s = postProcessing.apply( i );
            int margin =  s.length() - pad;
            if ( margin > 0 ) s = s.substring( 0, pad - 2 ) + "..";
            return s;
        };
        else finalProcessing = postProcessing;

        return finalProcessing;
    }

    public String toString() {
        return toString("");
    }

    public String toString( String deep )
    {
        if ( _tensor.isEmpty() ) return "empty";
        else if ( _tensor.isUndefined() ) return "undefined";
        _asStr = new StringBuilder();
        String base = ( deep == null ) ? "" : "\n" + deep;
        String delimiter = ( deep == null ) ? "" : "    ";
        String half = ( deep == null ) ? "" : "  ";
        String deeper = ( deep == null ) ? deep : deep + delimiter;
        if ( _hasShape ) _strShape();
        if ( !_hasValue ) return _asStr.toString();
        _$( ":" );
        if ( _isFormatted ) _format( new int[ _tensor.rank() ], -1 );
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
        if ( _hasRecursiveGraph && _tensor.has( GraphNode.class ) && _tensor.find( GraphNode.class ).size() > 0 ) {
            GraphNode<?> node = _tensor.find( GraphNode.class );
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
        if ( _hasDerivatives && _tensor.has( GraphNode.class ) && _tensor.find( GraphNode.class ).size() > 0 ) {
            GraphNode<?> node = _tensor.find( GraphNode.class );
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
        IntFunction<String> getter = _createValStringifier( _tensor.getData() );
        int size = _tensor.size();
        int trim = ( size - max );
        size = ( trim > 0 ) ? max : size;
        for ( int i = 0; i < size; i++ ) {
            String vStr = getter.apply( ( _tensor.isVirtual() ) ? 0 : _tensor.i_of_i( i ) );
            _$( vStr );
            if ( i < size - 1 ) _$( ", " );
            else if ( trim > 0 ) _$( ", ... + " )._$( trim )._$( " more" );
        }
    }

    //::: formatted...

    private void _buildRow(
            int size, int trimStart, int trimEnd, int trim, int[] idx, Function<int[],String> getter, String delimiter
    ) {
        for ( int i = 0; i < size; i++ ) {
            if ( i < trimStart || i >= trimEnd ) {
                _$( getter.apply( idx ) );
                if ( i < size - 1 ) _$( delimiter );
            }
            else if ( i == trimStart ) _$( "... " )._$( trim )._$( " more ..., " );

            NDConfiguration.Utility.increment( idx, _shape );
        }
    }


    // A recursive stringifier for formatted tensors...
    private void _format( int[] idx, int depth )
    {
        int max = ( _shortage * 32 / 50 );
        depth = ( depth < 0 ) ? 0 : depth;
        int size = _shape[ depth ];
        int trim = ( size - max );
        trim = Math.max( trim, 0 );
        int trimStart = ( size / 2 - trim / 2 );
        int trimEnd = ( size / 2 + trim / 2 );
        assert trimEnd - trimStart == trim;
        IndexAlias alias = _tensor.find( IndexAlias.class );
        if ( depth == idx.length - 1 ) {
            if (
                    alias != null &&
                            idx[ idx.length - 1 ] == 0 &&
                            idx[ Math.max( idx.length - 2, 0 ) ] == 0
            ) {
                List<Object> key = alias.keysOf( idx.length - 1 );
                if ( key != null ) {
                    _$( Util.indent( depth ) );
                    _$( (_legacy) ? "[ " : "( " ); // The following assert has prevented many String miscarriages!
                    assert key.size() - _shape[ idx.length - 1 ] == 0; // This is a basic requirement for the label size...
                    IntFunction<String> getter = _createValStringifier( key.toArray() );
                    _buildRow(
                            size, trimStart, trimEnd, trim,
                            new int[ idx.length ],
                            iarr -> getter.apply( iarr[ iarr.length -1 ] ),
                            (_legacy) ? "][" : ")("
                    );
                    _$( (_legacy) ? " ]" : " )" );
                    if ( alias.getTensorName() != null )
                        _$( (_legacy) ? ":[ " : ":( " )._$( alias.getTensorName() )._$( (_legacy) ? " ]" : " )" );
                    _$( "\n" );
                }
            }
            _$( Util.indent( depth ) );
            _$( (_legacy) ? "( " : "[ " );
            IntFunction<String> getter = _createValStringifier( _tensor.getData() );
            Function<int[], String> fun = ( _tensor.isVirtual() )
                    ? iarr -> getter.apply( 0 )
                    : iarr -> getter.apply( _tensor.i_of_idx( iarr ) );

            _buildRow( size, trimStart, trimEnd, trim, idx, fun, ", " );
            _$( (_legacy) ? " )" : " ]" );

            if ( alias != null ) _$( ":" )._buildSingleLabel( alias, depth, idx );
        } else {
            _$( Util.indent( depth ) );
            if ( depth > 0 && alias != null )
                _buildSingleLabel( alias, depth, idx )._$(":");
            _$( (_legacy) ? "(\n" : "[\n" );
            int i = 0;
            do {
                if ( i < trimStart || i >= trimEnd )
                    _format( idx, depth + 1 );
                else if ( i == trimStart )
                    _$( Util.indent( depth + 1 ) )._$( "... " )._$( trim )._$( " more ...\n" );
                else
                    idx[ depth ] = trimEnd + 1; // Jumping over trimmed entries!
                i++;
            }
            while ( idx[ depth ] != 0 );
            _$( Util.indent( depth ) )._$( (_legacy) ? ")" : "]" );
        }
        int i = depth - 1;
        if ( i >= 0 && i < idx.length && idx[ i ] != 0 ) _$( "," );
        _$( "\n" );
    }

    private TsrAsString _buildSingleLabel( IndexAlias alias, int depth, int[] idx ) {
        int pos = depth - 1;
        List<Object> key = alias.keysOf( pos );
        if ( pos >= 0 && key != null ) {
            _$( (_legacy) ? "[ " : "( ");
            int i = ( depth == idx.length - 1 )
                    ? ( _shape[ pos ] + idx[ pos ] - 1 ) % _shape[ pos ]
                    : idx[ pos ];
            _$( key.get( i ).toString() );
            _$( (_legacy) ? " ]" : " )");
        }
        return this;
    }

    private void _strShape() {
        boolean legacy = Neureka.instance().settings().view().isUsingLegacyView();
        _$( (legacy) ? "[" : "(" );
        for ( int i = 0; i < _shape.length; i++ ) {
            _$( _shape[ i ] );
            if ( i < _shape.length - 1 ) _$( "x" );
        }
        _$( (legacy) ? "]" : ")" );
    }

    public static class Util
    {
        public static String indent( int n ){
            return String.join("", Collections.nCopies( n, "   " ));
        }

        public static String pad( int left, String s ) {
            return String.join("", Collections.nCopies( left, " " )) + s;
        }

        public static String pad( String s, int right ) {
            return s + String.join("", Collections.nCopies( right, " " ));
        }

        @Contract( pure = true )
        public static String formatFP( double v )
        {
            DecimalFormatSymbols formatSymbols = new DecimalFormatSymbols( Locale.US );
            DecimalFormat Formatter = new DecimalFormat("##0.0##E0", formatSymbols);
            String vStr = String.valueOf( v );
            final int offset = 0;
            if ( vStr.length() > ( 7 - offset ) ) {
                if ( vStr.startsWith("0.") ) {
                    vStr = vStr.substring( 0, 7-offset )+"E0";
                } else if ( vStr.startsWith( "-0." ) ) {
                    vStr = vStr.substring( 0, 8-offset )+"E0";
                } else {
                    vStr = Formatter.format( v );
                    vStr = (!vStr.contains(".0E0"))?vStr:vStr.replace(".0E0",".0");
                    vStr = (vStr.contains("."))?vStr:vStr.replace("E0",".0");
                }
            }
            return vStr;
        }

    }

}
