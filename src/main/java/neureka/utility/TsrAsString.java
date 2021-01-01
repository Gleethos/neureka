package neureka.utility;

import lombok.Getter;
import lombok.experimental.Accessors;
import neureka.Neureka;
import neureka.Tsr;
import neureka.autograd.GraphNode;
import neureka.ndim.AbstractNDArray;
import neureka.ndim.config.NDConfiguration;

import java.util.Map;
import java.util.concurrent.atomic.AtomicReference;
import java.util.function.IntFunction;

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
        HAS_DERIVATIVES,
        HAS_RECURSIVE_GRAPH,

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

    private Tsr<?> _tensor;

    private Map<Should, Object> _config;

    public TsrAsString( Tsr<?> tensor, Map< Should, Object > settings ) {
        _construct( tensor, settings );
    }

    public TsrAsString( Tsr<?> tensor, String modes ) {
        _construct(
                tensor,
                Map.of(
                        Should.BE_SHORTENED_BY, (modes.contains("s")) ? 3 : 50,
                        Should.BE_COMPACT, modes.contains("c"),
                        Should.BE_FORMATTED, modes.contains("f"),
                        Should.HAVE_GRADIENT, modes.contains("g"),
                        Should.HAVE_PADDING_OF, (modes.contains("p")) ? 6 : -1,
                        Should.HAVE_VALUE, !(modes.contains("shp") || modes.contains("shape")),
                        Should.HAS_RECURSIVE_GRAPH, modes.contains("r"),
                        Should.HAS_DERIVATIVES, modes.contains("d")
                )
        );
    }

    private void _construct( Tsr tensor, Map< Should, Object > settings )
    {
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

        if ( settings.containsKey( Should.HAS_RECURSIVE_GRAPH ) )
            _hasRecursiveGraph = (boolean) settings.get( Should.HAS_RECURSIVE_GRAPH );

        if ( settings.containsKey( Should.HAS_DERIVATIVES ) )
            _hasDerivatives = (boolean) settings.get( Should.HAS_DERIVATIVES );
    }

    private IntFunction<String> _createValStringifier() {
        boolean compact = _isCompact;
        int pad = _padding;
        Object v = _tensor.getData();
        IntFunction<String> function;
        if ( v instanceof double[] )
            function = i -> ( compact )
                    ? AbstractNDArray.Utility.Stringify.formatFP( ( (double[]) v )[ i ])
                    : String.valueOf( ( (double[] ) v )[ i ] );
        else if ( v instanceof float[] )
            function = i -> ( compact )
                    ? AbstractNDArray.Utility.Stringify.formatFP( ( (float[]) v )[ i ] )
                    : String.valueOf( ( (float[]) v )[ i ] );
        else if ( v instanceof short[] )
            function = i -> ( compact )
                    ? AbstractNDArray.Utility.Stringify.formatFP( ( (short[]) v )[ i ] )
                    : String.valueOf( ( (short[]) v )[ i ] );
        else if ( v instanceof int[] )
            function = i -> ( compact )
                    ? AbstractNDArray.Utility.Stringify.formatFP( ( (int[]) v )[ i ] )
                    : String.valueOf( ( (int[]) v )[ i ] );
        else if ( v == null )
            function = i -> ( compact )
                    ? AbstractNDArray.Utility.Stringify.formatFP( _tensor.value64( i ) )
                    : String.valueOf( _tensor.value64( i ) );
        else
            function = i -> String.valueOf( ( (Object[]) v )[ i ] );

        if ( pad < 3 )
            return function;
        else return i -> {
            String s = function.apply( i );
            int margin = pad - s.length();
            int right = ( margin % 2 == 0 ) ? margin / 2 : ( margin-1 ) / 2;
            if ( margin > 0 ) s = ( " ".repeat( margin-right ) + s + " ".repeat( right ) );
            return s;
        };
    }

    public String toString( String deep )
    {
        if ( _tensor.isEmpty() ) return "empty";
        else if ( _tensor.isUndefined() ) return "undefined";

        String base = ( deep == null ) ? "" : "\n" + deep;
        String delimiter = ( deep == null ) ? "" : "    ";
        String half = ( deep == null ) ? "" : "  ";
        String deeper = ( deep == null ) ? deep : deep + delimiter;
        StringBuilder strShape = _strShape();
        if ( !_hasValue ) return strShape.toString();
        String asString = strShape.toString();
        if ( _isFormatted )
            asString += ":" + _format( _tensor.getNDConf().shape(), new int[ _tensor.rank() ], -1 ).toString();
        else
            asString +=  (
                    ( Neureka.instance().settings().view().isUsingLegacyView() )
                            ? ":(" + _stringified() + ")"
                            : ":[" + _stringified() + "]"
            );
        if ( _hasGradient && ( _tensor.rqsGradient() || _tensor.hasGradient() ) ) {
            asString += ":g:";
            Tsr<?> gradient = _tensor.find( Tsr.class );
            if ( gradient != null )
                asString += gradient
                        .toString( "c" ).replace( strShape + ":", "" );
            else
                asString += ( (Neureka.instance().settings().view().isUsingLegacyView() ) ? "(null)" : "[null]" );
        }
        if ( _hasRecursiveGraph && _tensor.has( GraphNode.class ) && _tensor.find( GraphNode.class ).size() > 0 ) {
            GraphNode<?> node = _tensor.find( GraphNode.class );
            AtomicReference<String> enclosed = new AtomicReference<>( "; " );
            node.forEachDerivative( ( t, agent ) -> {
                if ( agent.derivative() == null ) {
                    enclosed.set( enclosed.get() + "->d(null), " );
                } else {
                    enclosed.set(
                            enclosed.get() +
                                    base + "=>d|[ " +
                                    base + delimiter + agent.derivative().toString( _config, deeper ) + " " +
                                    base + half + "]|:t{ " +
                                    base + delimiter + (
                                    ( t.getPayload() != null ) ? t.getPayload().toString( _config, deeper ) : t.toString("")
                            ) + " " +
                                    base + half + "}, "
                    );
                }
            });
            asString += enclosed.get();
        }
        if ( _hasDerivatives && _tensor.has( GraphNode.class ) && _tensor.find( GraphNode.class ).size() > 0 ) {
            GraphNode<?> node = _tensor.find( GraphNode.class );
            if ( node.getMode() != 0 ) {
                AtomicReference<String> asAR = new AtomicReference<>( "; " );
                node.forEachDerivative( ( t, agent ) -> {
                    if ( agent.derivative() == null ) asAR.set( asAR.get() + "->d(" + agent.toString() + "), " );
                    else asAR.set( asAR.get() + "->d" + agent.derivative().toString( _config, deeper ) + ", " );
                });
                asString += asAR.get();
            }
        }
        return asString;

    }

    private String _stringified() {
        int max = _shortage;
        IntFunction<String> getter = _createValStringifier();
        StringBuilder asString = new StringBuilder();
        int size = _tensor.size();
        int trim = ( size - max );
        size = ( trim > 0 ) ? max : size;
        for ( int i = 0; i < size; i++ ) {
            String vStr = getter.apply( ( _tensor.isVirtual() ) ? 0 : _tensor.i_of_i( i ) );
            asString.append( vStr );
            if ( i < size - 1 ) asString.append( ", " );
            else if ( trim > 0 ) asString.append( ", ... + " ).append( trim ).append( " more" );
        }
        return asString.toString();
    }

    //::: formatted...

    private String _stringified(
            int max,
            int[] idx,
            int size
    ) {
        int[] shape = _tensor.getNDConf().shape();
        IntFunction<String> getter = _createValStringifier();
        StringBuilder asString = new StringBuilder();
        int trim = ( size - max );
        size = ( trim > 0 ) ? max : size;
        for ( int i = 0; i < size; i++ ) {
            String vStr = getter.apply( ( _tensor.isVirtual() ) ? 0 : _tensor.i_of_idx( idx ) );
            asString.append( vStr );
            if ( i < size - 1 ) asString.append( ", " );
            else if ( trim > 0 ) asString.append( ", ... + " ).append( trim ).append( " more" );
            NDConfiguration.Utility.increment( idx, shape );
        }
        for ( int i = 0; i < trim; i++ ) NDConfiguration.Utility.increment( idx, shape );
        return asString.toString();
    }

    // A recursive stringifier for formatted tensors...
    private StringBuilder _format(int[] shape, int[] idx, int dim )
    {
        boolean legacy = Neureka.instance().settings().indexing().isUsingLegacyIndexing();
        dim = ( dim < 0 ) ? ( (!legacy) ? 0 : _tensor.rank()-1 ) : dim;
        int depth = ( !legacy ) ? dim : _tensor.rank()-dim-1 ;
        StringBuilder builder = new StringBuilder();

        if ( legacy && dim == 0 || !legacy && dim == idx.length - 1 ) {
            builder.append( "   ".repeat( depth ) );
            builder.append( "[ " + _stringified( 50, idx, shape[ dim ] ) + " ]" );
        } else {
            builder.append( "   ".repeat(depth) + "[" );
            builder.append( "\n" );
            do {
                builder.append( _format( shape, idx, dim + ( (!legacy) ? 1 : -1 ) ) );
            } while( idx[ dim ] != 0 );
            builder.append("   ".repeat(depth) + "]");
        }
        int i = dim + ( (legacy) ? 1 : -1 );
        if ( i >= 0 && i < idx.length && idx[ i ] != 0 ) builder.append( "," );
        builder.append( "\n" );
        return builder;
    }

    private StringBuilder _strShape() {
        boolean legacy = Neureka.instance().settings().view().isUsingLegacyView();
        StringBuilder strShape = new StringBuilder( (legacy) ? "[" : "(" );
        int[] shape = _tensor.getNDConf().shape();
        for ( int i = 0; i < shape.length; i++ ) {
            strShape.append( shape[ i ] );
            if ( i < shape.length - 1 ) strShape.append( "x" );
        }
        if ( legacy ) return strShape.append( "]" );
        else return strShape.append( ")" );
    }

}
