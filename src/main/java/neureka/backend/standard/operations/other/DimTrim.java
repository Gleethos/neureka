package neureka.backend.standard.operations.other;

import neureka.Neureka;
import neureka.Tsr;
import neureka.autograd.DefaultADAgent;
import neureka.calculus.args.Arg;
import neureka.backend.api.ExecutionCall;
import neureka.backend.api.operations.AbstractOperation;
import neureka.backend.api.operations.OperationBuilder;
import neureka.backend.standard.algorithms.GenericAlgorithm;
import neureka.calculus.Function;
import neureka.calculus.assembly.FunctionBuilder;
import neureka.devices.Device;
import neureka.ndim.config.AbstractNDC;

import java.util.ArrayList;
import java.util.List;

public class DimTrim extends AbstractOperation
{

    public DimTrim()
    {
        super(
                new OperationBuilder()
                        .setFunction(         "dimtrim"   )
                        .setOperator(         "dimtrim"   )
                        .setArity(            1           )
                        .setIsOperator(       false       )
                        .setIsIndexer(        false       )
                        .setIsDifferentiable( true        )
                        .setIsInline(         false       )
        );

        GenericAlgorithm implementation = new GenericAlgorithm("reshape")
                .setIsSuitableFor( call -> 1.0f )
                .setCanPerformBackwardADFor( call -> true )
                .setCanPerformForwardADFor( call -> false )
                .setSupplyADAgentFor(
                        ( Function f, ExecutionCall<? extends Device<?>> call, boolean forward ) ->
                        {
                            int prefix = call.find(Arg.Ends.class).get()[ 0 ];
                            int postfix = call.find(Arg.Ends.class).get()[ 1 ];
                            if ( forward ) {
                                throw new IllegalArgumentException("Dim-Trim operation does not support forward-AD!");
                            }
                            return new DefaultADAgent()
                                    .withContext(call.findAll(Arg.class))
                                    .setForward((t, derivative) -> new FunctionBuilder(Neureka.get().context()).build(f.toString(), false).derive(new Tsr[]{derivative},0))
                                    .setBackward( (t, error) -> pad(error, new int[]{prefix, postfix}, true) );
                        }
                )
                .setHandleInsteadOfDevice(
                        ( caller, call ) ->
                        {
                            Tsr<?>[] inputs = caller.srcActivation(call.getTensors(), call.getJ(), -1, 0);
                            assert inputs.length == 1;
                            Tsr<?> t = inputs[ 0 ];
                            if ( call.getDerivativeIndex() == 0 ) {
                                int prefix = call.find(Arg.Ends.class).get()[ 0 ];
                                int postfix = call.find(Arg.Ends.class).get()[ 0 ];
                                return pad(t, new int[]{prefix, postfix}, true);
                            } else {
                                int[] ends = new int[ 2 ];
                                call.set(Arg.Ends.of(ends));
                                return trim(t, ends, true);
                            }
                        }
                )
                .setHandleRecursivelyAccordingToArity( (call, goDeeperWith ) -> null )
                .setInstantiateNewTensorsForExecutionIn( call -> call )
                .build();

        setAlgorithm(
                GenericAlgorithm.class,
                implementation
        );

    }

    public static Tsr pad(Tsr tensor, int[] ends, boolean newTsr) {
        tensor = (newTsr) ? (Tsr)tensor.getAt(new ArrayList<>()) : tensor;
        List<Integer> newShape = new ArrayList<>();
        List<Integer> newTranslation = new ArrayList<>();
        List<Integer> newIdxmap = new ArrayList<>();
        List<Integer> newSpread = new ArrayList<>();
        List<Integer> newOffset = new ArrayList<>();
        int[] shape = tensor.getNDConf().shape();
        int prefix = ends[ 0 ];
        int postfix = ends[ 1 ];
        for ( int i = 0; i < prefix; i++ ) {
            newShape.add( 1 );
            newTranslation.add( 1 );
            newIdxmap.add( 1 );
            newSpread.add( 0 );
            newOffset.add( 0 );
        }
        for ( int i = 0; i < shape.length; i++ ) {
            newShape.add(shape[ i ]);
            newTranslation.add(tensor.getNDConf().translation( i ));
            newIdxmap.add(tensor.getNDConf().indicesMap( i ));
            newSpread.add(tensor.getNDConf().spread( i ));
            newOffset.add(tensor.getNDConf().offset( i ));
        }
        for ( int i = 0; i < postfix; i++ ) {
            newShape.add( 1 );
            newTranslation.add( 1 );
            newIdxmap.add( 1 );
            newSpread.add( 0 );
            newOffset.add( 0 );
        }
        tensor.setNDConf(
                AbstractNDC.construct(
                        newShape.stream().mapToInt(i->i).toArray(),
                        newTranslation.stream().mapToInt(i->i).toArray(),
                        newIdxmap.stream().mapToInt(i->i).toArray(),
                        newSpread.stream().mapToInt(i->i).toArray(),
                        newOffset.stream().mapToInt(i->i).toArray()
                )
        );
        return tensor;
    }

    public static Tsr<?> trim(Tsr<?> tensor, int[] ends, boolean newTsr)
    {
        tensor = (newTsr) ? (Tsr<?>)tensor.getAt(new ArrayList<>()) : tensor;
        List<Integer> newShape = new ArrayList<>();
        List<Integer> newTranslation = new ArrayList<>();
        List<Integer> newIdxmap = new ArrayList<>();
        List<Integer> newSpread = new ArrayList<>();
        List<Integer> newOffset = new ArrayList<>();
        int[] shape = tensor.getNDConf().shape();
        int prefix = 0;
        for ( int s : shape) if (s == 1) prefix++; else break;
        int postfix = 0;
        for ( int i=shape.length-1; i>=0; i-- ) if ( shape[ i ] == 1 ) postfix++; else break;
        for ( int i = prefix; i < shape.length-postfix; i++ ) {
            newShape.add(shape[ i ]);
            newTranslation.add(tensor.getNDConf().translation( i ));
            newIdxmap.add(tensor.getNDConf().indicesMap( i ));
            newSpread.add(tensor.getNDConf().spread( i ));
            newOffset.add(tensor.getNDConf().offset( i ));
        }
        tensor.setNDConf(
                AbstractNDC.construct(
                        newShape.stream().mapToInt(i->i).toArray(),
                        newTranslation.stream().mapToInt(i->i).toArray(),
                        newIdxmap.stream().mapToInt(i->i).toArray(),
                        newSpread.stream().mapToInt(i->i).toArray(),
                        newOffset.stream().mapToInt(i->i).toArray()
                )
        );
        ends[ 0 ] = prefix;
        ends[ 1 ] = postfix;
        return tensor;
    }


    @Override
    public String stringify( String[] children ) {
        String expression = String.join( ", ", children );
        if (expression.charAt( 0 ) == '(' && expression.charAt( expression.length() - 1 ) == ')') {
            return "dimtrim" + expression;
        }
        return "dimtrim" + "(" + expression + ")";
    }

    @Override
    public String asDerivative( Function[] children, int derivationIndex) {
        throw new IllegalStateException("Operation does not support dynamic derivation!");
    }

    @Override
    public double calculate( double[] inputs, int j, int d, Function[] src ) {
        return src[ 0 ].call( inputs, j );
    }
}
