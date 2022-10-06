package neureka.backend.main.operations.other;

import neureka.Neureka;
import neureka.backend.api.AutoDiffMode;
import neureka.backend.api.ExecutionCall;
import neureka.backend.api.fun.SuitabilityPredicate;
import neureka.backend.api.template.algorithms.AbstractDeviceAlgorithm;
import neureka.backend.api.template.operations.AbstractOperation;
import neureka.backend.api.template.operations.OperationBuilder;
import neureka.backend.main.algorithms.Activation;
import neureka.backend.main.algorithms.Scalarization;
import neureka.calculus.Function;
import neureka.calculus.args.Arg;

public class AssignLeft extends AbstractOperation
{
    public AssignLeft() {
        super(
            new OperationBuilder()
                .identifier(       "left_inline"  )
                .operator(         "<"            )
                .arity(            -2             )
                .isOperator(       true           )
                .isIndexer(        false          )
                .isDifferentiable( false          )
                .isInline(         true           )
        );

        setAlgorithm(
            Scalarization.class,
            new Scalarization()
            .setIsSuitableFor(
               call -> {
                   int offset = ( call.arity() == 1 ? 0 : 1 );
                   if ( call.input( offset ).isVirtual() || call.input( offset ).size() == 1 )
                       return SuitabilityPredicate.GOOD;
                   else
                       return SuitabilityPredicate.UNSUITABLE;
               }
            )
            .setAutogradModeFor( call -> AutoDiffMode.NOT_SUPPORTED)
            .setDeviceExecution( (call, callback) -> AbstractDeviceAlgorithm.executeDeviceAlgorithm( call, callback ) )
            .setCallPreparation(
                call -> {
                    int offset = ( call.input( 0 ) == null ? 1 : 0 );
                    call.input( offset ).getMut().incrementVersion(call);
                    call.input( offset ).getMut().setIsVirtual( false );
                    return
                        ExecutionCall.of( call.input( offset ), call.input( 1+offset ) )
                                .andArgs(Arg.DerivIdx.of(-1))
                                .running(this)
                                .on( call.getDevice() );
                }
            )
            .buildFunAlgorithm()
        );

        setAlgorithm(
            new Activation()
            .setIsSuitableFor(
                call -> call.validate()
                        .allNotNull( t -> t.getDataType().typeClassImplements(Object.class) )
                        .allNotNull( t -> !t.isVirtual() )
                        .tensors( tensors -> tensors.length == 2 || tensors.length == 3 )
                        .suitabilityIfValid(SuitabilityPredicate.EXCELLENT)
            )
            .setAutogradModeFor( call -> AutoDiffMode.NOT_SUPPORTED)
            .setDeviceExecution( (call, callback) -> AbstractDeviceAlgorithm.executeDeviceAlgorithm( call, callback ) )
            .setCallPreparation(
                    call -> {
                        int offset = ( call.input( 0 ) == null ? 1 : 0 );
                        call.input( offset ).getMut().incrementVersion(call);
                        return ExecutionCall.of( call.input(offset), call.input(1+offset) )
                                .running(Neureka.get().backend().getOperation("idy"))
                                .on( call.getDevice() );
                    }
            )
            .buildFunAlgorithm()
        );
    }


    @Override
    public String stringify( String[] children ) {
        StringBuilder reconstructed = new StringBuilder();
        for ( int i = 0; i < children.length; ++i ) {
            reconstructed.append( children[ i ] );
            if ( i < children.length - 1 ) reconstructed.append(" <- ");
        }
        return "(" + reconstructed + ")";
    }

    @Override
    public double calculate( double[] inputs, int j, int d, Function[] src ) {
            return src[ 0 ].call( inputs, j );
    }
}
