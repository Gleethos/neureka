package neureka.backend.standard.operations.other;

import neureka.Neureka;
import neureka.Tsr;
import neureka.backend.api.ExecutionCall;
import neureka.backend.api.algorithms.fun.AutoDiffMode;
import neureka.backend.api.algorithms.fun.Result;
import neureka.backend.api.algorithms.fun.SuitabilityPredicate;
import neureka.backend.api.operations.AbstractOperation;
import neureka.backend.api.operations.OperationBuilder;
import neureka.backend.standard.algorithms.Activation;
import neureka.backend.standard.algorithms.Scalarization;
import neureka.backend.standard.algorithms.internal.Fun;
import neureka.backend.standard.implementations.CLImplementation;
import neureka.backend.standard.implementations.CPUImplementation;
import neureka.calculus.Function;
import neureka.calculus.args.Arg;
import neureka.calculus.internal.CalcUtil;
import neureka.devices.host.CPU;
import neureka.devices.opencl.OpenCLDevice;

public class AssignLeft extends AbstractOperation
{
    public AssignLeft() {
        super(
            new OperationBuilder()
            .setIdentifier(       "left_inline"  )
            .setOperator(         "<"            )
            .setArity(            -2             )
            .setIsOperator(       true           )
            .setIsIndexer(        false          )
            .setIsDifferentiable( false          )
            .setIsInline(         true           )
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
            .setExecution(
                    (caller, call) ->
                            Result.of(CalcUtil.defaultRecursiveExecution(caller, call)).withAutoDiff(getDefaultAlgorithm())
            )
            .setCallPreparation(
                call -> {
                    int offset = ( call.input( 0 ) == null ? 1 : 0 );
                    call.input( offset ).getUnsafe().incrementVersion(call);
                    call.input( offset ).setIsVirtual( false );
                    return
                        ExecutionCall.of( call.input( offset ), call.input( 1+offset ) )
                                .andArgs(Arg.DerivIdx.of(-1))
                                .running(this)
                                .on( call.getDevice() );
                }
            )
            .buildFunAlgorithm()
            .setImplementationFor(
                CPU.class,
                Scalarization.implementationForCPU()
                    .with(Fun.F64F64ToF64.of( ( a, b ) -> b ))
                    .with(Fun.F32F32ToF32.of( ( a, b ) -> b ))
                    .with(Fun.F32F32ToF32.of( ( a, b ) -> b ))
                    .with(Fun.ObjObjToObj.of( ( a, b ) -> b ))
                    .get()
            )
            .setImplementationFor(
                OpenCLDevice.class,
                CLImplementation
                    .compiler()
                    .arity( 2 )
                    .kernelSource( Scalarization.getKernelSource() )
                    .activationSource( "output = value;\n" )
                    .differentiationSource( "output = value;\n" )
                    .kernelPostfix( this.getIdentifier() )
                    .execution(
                        call -> {
                            Tsr<Number> t = call.input( Number.class, 0 );
                            int gwz = t.size();
                            call.getDevice()
                                .getKernel(call)
                                .passAllOf( t )
                                .passAllOf( t )
                                .pass( call.input( Number.class, 1 ).at(0).get().floatValue() )
                                .pass( t.rank() )
                                .pass( call.getValOf( Arg.DerivIdx.class ) )
                                .call( gwz );
                        }
                    )
                    .build()
            )
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
            .setExecution(
                    (caller, call) ->
                            Result.of(CalcUtil.defaultRecursiveExecution(caller, call)).withAutoDiff(getDefaultAlgorithm())
            )
            .setCallPreparation(
                    call -> {
                        int offset = ( call.input( 0 ) == null ? 1 : 0 );
                        call.input( offset ).getUnsafe().incrementVersion(call);
                        return ExecutionCall.of( call.input(offset), call.input(1+offset) )
                                .running(Neureka.get().backend().getOperation("idy"))
                                .on( call.getDevice() );
                    }
            )
            .buildFunAlgorithm()
            .setImplementationFor(
                CPU.class,
                CPUImplementation
                    .withArity(2)
                    .andImplementation(
                        call -> {
                            call.input( 0 ).setIsVirtual( false );
                            Neureka.get().backend().getOperation("idy")
                                    .getAlgorithm( Activation.class )
                                    .getImplementationFor( CPU.class )
                                    .run(call);
                        }
                    )
            )
            .setImplementationFor(
                OpenCLDevice.class,
                Activation.implementationForGPU( this.getIdentifier() )
                        .with( "output = input;\n" )
                        .and( "output = input;\n" )
            )
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
    public String asDerivative( Function[] children, int derivationIndex) {
        throw new IllegalStateException("Operation does not support dynamic derivation!");
    }

    @Override
    public double calculate( double[] inputs, int j, int d, Function[] src ) {
            return src[ 0 ].call( inputs, j );
    }
}
