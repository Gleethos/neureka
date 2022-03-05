package neureka.backend.standard.operations.other;

import neureka.Neureka;
import neureka.Tsr;
import neureka.backend.api.ExecutionCall;
import neureka.backend.standard.algorithms.internal.Fun;
import neureka.backend.api.algorithms.fun.SuitabilityPredicate;
import neureka.backend.api.operations.AbstractOperation;
import neureka.backend.api.operations.OperationBuilder;
import neureka.backend.standard.algorithms.Activation;
import neureka.backend.standard.algorithms.Scalarization;
import neureka.backend.standard.implementations.CLImplementation;
import neureka.backend.standard.implementations.CPUImplementation;
import neureka.calculus.internal.CalcUtil;
import neureka.calculus.Function;
import neureka.calculus.args.Arg;
import neureka.devices.host.CPU;
import neureka.devices.opencl.OpenCLDevice;

public class AssignLeft extends AbstractOperation
{
    public AssignLeft() {
        super(
            new OperationBuilder()
                    .setFunction(         "left_inline"  )
                    .setOperator(         "<"            )
                    .setArity(            -2             )
                    .setIsOperator(       true           )
                    .setIsIndexer(        false          )
                    .setIsDifferentiable( false          )
                    .setIsInline(         true           )
        );

        Scalarization scalarization = new Scalarization()
                .setIsSuitableFor(
                    call -> {
                        if ( call.tensor( 1 ).isVirtual() || call.tensor( 1 ).size() == 1 )
                            return SuitabilityPredicate.GOOD;
                        else
                            return SuitabilityPredicate.UNSUITABLE;
                    }
                )
                .setCanPerformBackwardADFor( call -> false )
                .setCanPerformForwardADFor( call -> false )
                .setSupplyADAgentFor( getDefaultAlgorithm() )
                .setExecutionDispatcher( CalcUtil::defaultRecursiveExecution)
                .setCallPreparation(
                    call ->
                    {
                        int offset = ( call.tensor( 0 ) == null ? 1 : 0 );
                        call.tensor( offset ).getUnsafe().incrementVersion(call);
                        call.tensor( offset ).setIsVirtual( false );
                        return
                            ExecutionCall.of( call.tensor( offset ), call.tensor( 1+offset ) )
                                            .andArgs(Arg.DerivIdx.of(-1))
                                            .running(this)
                                            .on( call.getDevice() );
                    }
                )
                .buildFunAlgorithm();

        setAlgorithm(
            Scalarization.class,
            scalarization.setImplementationFor(
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
                    .kernelSource( scalarization.getKernelSource() )
                    .activationSource( "output = value;\n" )
                    .differentiationSource( "output = value;\n" )
                    .kernelPostfix( this.getFunction() )
                    .execution(
                        call -> {
                            Tsr<Number> t = call.getTsrOfType( Number.class, 0 );
                            int gwz = t.size();
                            call.getDevice()
                                .getKernel(call)
                                .passAllOf( t )
                                .passAllOf( t )
                                .pass( call.getTsrOfType( Number.class, 1 ).getValueAs( float[].class )[ 0 ] )
                                .pass( t.rank() )
                                .pass( call.getValOf( Arg.DerivIdx.class ) )
                                .call( gwz );
                        }
                    )
                    .build()
            )
        );

        Activation activation = new Activation()
            .setIsSuitableFor(
                    call -> call.validate()
                            .allNotNull( t -> t.getDataType().typeClassImplements(Object.class) )
                            .basicSuitability()
            )
            .setCanPerformBackwardADFor( call -> false )
            .setCanPerformForwardADFor( call -> false )
            .setSupplyADAgentFor( getDefaultAlgorithm() )
            .setExecutionDispatcher( CalcUtil::defaultRecursiveExecution )
            .setCallPreparation(
                call -> {
                    int offset = ( call.tensor( 0 ) == null ) ? 1 : 0;
                    call.tensor( offset ).getUnsafe().incrementVersion(call);
                    return ExecutionCall.of( call.tensor(offset), call.tensor(1+offset) )
                                        .andArgs(Arg.DerivIdx.of(-1))
                                        .running(Neureka.get().backend().getOperation("idy"))
                                        .on( call.getDevice() );
                }
            )
            .buildFunAlgorithm();

        setAlgorithm(
            Activation.class,
            activation
                .setImplementationFor(
                    CPU.class,
                    CPUImplementation
                        .withArity(2)
                        .andImplementation(
                            call -> {
                                call.tensor( 0 ).setIsVirtual( false );
                                Neureka.get().backend().getOperation("idy")
                                        .getAlgorithm( Activation.class )
                                        .getImplementationFor( CPU.class )
                                        .run(call);
                            }
                        )
                )
                .setImplementationFor(
                    OpenCLDevice.class,
                    Activation.implementationForGPU( this.getFunction() )
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
