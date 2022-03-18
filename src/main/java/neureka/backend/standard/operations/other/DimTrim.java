package neureka.backend.standard.operations.other;

import neureka.Neureka;
import neureka.Tsr;
import neureka.autograd.ADAgent;
import neureka.backend.api.Algorithm;
import neureka.backend.api.ExecutionCall;
import neureka.backend.api.algorithms.fun.SuitabilityPredicate;
import neureka.backend.api.operations.AbstractOperation;
import neureka.backend.api.operations.OperationBuilder;
import neureka.backend.standard.algorithms.FunAlgorithm;
import neureka.calculus.internal.CalcUtil;
import neureka.calculus.Function;
import neureka.calculus.args.Arg;
import neureka.calculus.assembly.FunctionBuilder;
import neureka.devices.Device;
import neureka.ndim.config.AbstractNDC;
import neureka.ndim.config.NDConfiguration;

import java.util.ArrayList;
import java.util.List;

public class DimTrim extends AbstractOperation
{

    public DimTrim()
    {
        super(
                new OperationBuilder()
                        .setIdentifier(         "dimtrim"   )
                        .setOperator(         "dimtrim"   )
                        .setArity(            1           )
                        .setIsOperator(       false       )
                        .setIsIndexer(        false       )
                        .setIsDifferentiable( true        )
                        .setIsInline(         false       )
        );
        setAlgorithm(
            FunAlgorithm.class,
            Algorithm
                .withName("dimTrim")
                .setIsSuitableFor( call -> SuitabilityPredicate.GOOD )
                .setCanPerformBackwardADFor( call -> true )
                .setCanPerformForwardADFor( call -> false )
                .setSupplyADAgentFor(
                    ( Function f, ExecutionCall<? extends Device<?>> call, boolean forward ) ->
                    {
                        int[] endings = endsFrom( call.input( 0 ).getNDConf().shape() );
                        int prefix  = endings[ 0 ];
                        int postfix = endings[ 1 ];
                        if ( forward )
                            throw new IllegalArgumentException("Dim-Trim operation does not support forward-AD!");

                        return ADAgent.of( null )
                                .withArgs( Arg.Ends.of(endings) )
                                .setForward(
                                      (t, derivative) ->
                                          new FunctionBuilder( Neureka.get().backend() )
                                                  .build(f.toString(), false)
                                                  .derive(new Tsr[]{derivative},0)
                                )
                                .setBackward( (t, error) -> _pad(error, new int[]{prefix, postfix}, true) );
                    }
                )
                .setExecutionDispatcher(
                    ( caller, call ) ->
                    {
                        Tsr<?>[] inputs = CalcUtil.srcActivation(
                                                call.inputs(), call.getValOf( Arg.VarIdx.class ), -1, 0,
                                                caller.getSubFunctions().toArray(new Function[0])
                                            );
                        assert inputs.length == 1;
                        Tsr<?> t = inputs[ 0 ];
                        if ( call.getValOf( Arg.DerivIdx.class ) == 0 ) {
                            int prefix = call.getValOf(Arg.Ends.class)[ 0 ];
                            int postfix = call.getValOf(Arg.Ends.class)[ 1 ];
                            return _pad( t, new int[]{prefix, postfix}, true );
                        } else
                            return _trim( t, true );
                    }
                )
                .setCallPreparation( call -> call )
                .buildFunAlgorithm()
        );
    }

    private static Tsr<?> _pad( Tsr<?> tensor, int[] ends, boolean newTsr ) {

        if ( tensor.getNDConf().getLayout() == NDConfiguration.Layout.COLUMN_MAJOR )
            throw new IllegalArgumentException("Column major not yet supported for shape trimming!");

        tensor = ( newTsr ? tensor.getAt(new ArrayList<>()) : tensor );
        List<Integer> newShape = new ArrayList<>();
        List<Integer> newTranslation = new ArrayList<>();
        List<Integer> newIndicesMap = new ArrayList<>();
        List<Integer> newSpread = new ArrayList<>();
        List<Integer> newOffset = new ArrayList<>();
        int[] shape = tensor.getNDConf().shape();
        int prefix = ends[ 0 ];
        int postfix = ends[ 1 ];
        for ( int i = 0; i < prefix; i++ ) {
            newShape.add( 1 );
            newTranslation.add( 1 );
            newIndicesMap.add( 1 );
            newSpread.add( 0 );
            newOffset.add( 0 );
        }
        for ( int i = 0; i < shape.length; i++ ) {
            newShape.add(shape[ i ]);
            newTranslation.add(tensor.getNDConf().translation( i ));
            newIndicesMap.add(tensor.getNDConf().indicesMap( i ));
            newSpread.add(tensor.getNDConf().spread( i ));
            newOffset.add(tensor.getNDConf().offset( i ));
        }
        for ( int i = 0; i < postfix; i++ ) {
            newShape.add( 1 );
            newTranslation.add( 1 );
            newIndicesMap.add( 1 );
            newSpread.add( 0 );
            newOffset.add( 0 );
        }
        tensor.getUnsafe()
              .setNDConf(
                    AbstractNDC.construct(
                             newShape.stream().mapToInt( i -> i ).toArray(),
                             newTranslation.stream().mapToInt( i -> i ).toArray(),
                             newIndicesMap.stream().mapToInt( i -> i ).toArray(),
                             newSpread.stream().mapToInt( i -> i ).toArray(),
                             newOffset.stream().mapToInt( i -> i ).toArray()
                     )
              );
        return tensor;
    }

    private static Tsr<?> _trim( Tsr<?> tensor, boolean newTsr )
    {
        if ( tensor.getNDConf().getLayout() == NDConfiguration.Layout.COLUMN_MAJOR )
            throw new IllegalArgumentException("Column major not yet supported for shape trimming!");

        tensor = ( newTsr ? tensor.getAt( new ArrayList<>() ).getUnsafe().setIsIntermediate( true ) : tensor );
        List<Integer> newShape = new ArrayList<>();
        List<Integer> newTranslation = new ArrayList<>();
        List<Integer> newIndicesMap = new ArrayList<>();
        List<Integer> newSpread = new ArrayList<>();
        List<Integer> newOffset = new ArrayList<>();
        int[] shape = tensor.getNDConf().shape();
        int[] endings = endsFrom( tensor.getNDConf().shape() );
        int prefix  = endings[ 0 ];
        int postfix = endings[ 1 ];

        for ( int i = prefix; i < shape.length-postfix; i++ ) {
            newShape.add( shape[ i ] );
            newTranslation.add( tensor.getNDConf().translation( i ) );
            newIndicesMap.add( tensor.getNDConf().indicesMap( i ) );
            newSpread.add( tensor.getNDConf().spread( i ) );
            newOffset.add( tensor.getNDConf().offset( i ) );
        }
        tensor.getUnsafe()
                  .setNDConf(
                     AbstractNDC.construct(
                         newShape.stream().mapToInt( i -> i ).toArray(),
                         newTranslation.stream().mapToInt( i -> i ).toArray(),
                         newIndicesMap.stream().mapToInt( i -> i ).toArray(),
                         newSpread.stream().mapToInt( i -> i ).toArray(),
                         newOffset.stream().mapToInt( i -> i ).toArray()
                     )
                  );

        return tensor;
    }

    public static int[] endsFrom( int[] shape ) {
        int prefix = 0;
        for ( int s : shape ) if ( s == 1 ) prefix++; else break;
        int postfix = 0;
        for ( int i = shape.length-1; i >= 0; i-- ) if ( shape[ i ] == 1 ) postfix++; else break;
        return new int[]{ prefix, postfix };
    }

    @Override
    public String stringify( String[] children ) {
        String expression = String.join( ", ", children );
        if ( expression.charAt( 0 ) == '(' && expression.charAt( expression.length() - 1 ) == ')' ) {
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
