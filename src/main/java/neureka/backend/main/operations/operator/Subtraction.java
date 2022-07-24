package neureka.backend.main.operations.operator;

import neureka.Tsr;
import neureka.autograd.ADAgent;
import neureka.backend.api.ExecutionCall;
import neureka.backend.api.AutoDiffMode;
import neureka.backend.api.fun.SuitabilityPredicate;
import neureka.backend.api.template.operations.AbstractOperation;
import neureka.backend.api.template.operations.OperationBuilder;
import neureka.backend.main.algorithms.Broadcast;
import neureka.backend.main.algorithms.Operator;
import neureka.backend.main.algorithms.Scalarization;
import neureka.backend.main.algorithms.internal.Fun;
import neureka.backend.main.implementations.CLImplementation;
import neureka.backend.main.operations.ElemWiseUtil;
import neureka.calculus.Function;
import neureka.calculus.args.Arg;
import neureka.devices.Device;
import neureka.devices.host.CPU;
import neureka.devices.opencl.OpenCLDevice;
import org.jetbrains.annotations.Contract;

import java.util.Arrays;
import java.util.stream.Collectors;

public class Subtraction extends AbstractOperation
{
    public Subtraction()
    {
        super(
                new OperationBuilder()
                        .setIdentifier(         "subtract"    )
                        .setOperator(         "-"        )
                        .setArity(            -1         )
                        .setIsOperator(       true       )
                        .setIsIndexer(        false      )
                        .setIsDifferentiable( true       )
                        .setIsInline(         false      )
        );

        //_____________________
        // DEFAULT OPERATION :

        Operator operator = new Operator(ElemWiseUtil::forSubtractions)
                                    .setSupplyADAgentFor( getDefaultAlgorithm() )
                                    .buildFunAlgorithm();
        setAlgorithm(
            operator.setImplementationFor(
                CPU.class,
                Operator.implementationForCPU()
                    .with(Fun.F64F64ToF64.triple(
                        ( a, b ) -> a - b,
                        ( a, b ) ->  1, // Deriving at input 0
                        ( a, b ) -> -1 // deriving input 1
                    ))
                    .with(Fun.F32F32ToF32.triple(
                        ( a, b ) -> a - b,
                        ( a, b ) ->  1, // Deriving at input 0
                        ( a, b ) -> -1 // deriving input 1
                    ))
                    .with(Fun.I32I32ToI32.triple(
                        ( a, b ) -> a - b,
                        ( a, b ) ->  1, // Deriving at input 0
                        ( a, b ) -> -1 // deriving input 1
                    ))
                    .get()
            )
            .setImplementationFor(
                OpenCLDevice.class,
                Operator.implementationForGPU( this.getIdentifier() )
                        .with( "output = input1 - input2;  \n" )
                        .and(
                                "if (d==0) {                  \n" +//drn and src2 switch:
                                        "    output = 1;              \n" +
                                        "} else {                     \n" +
                                        "    output = -1;               " +
                                        "}"
                        )
            )
        );

        //___________________________
        // TENSOR SCALAR OPERATION :

        Scalarization scalarization =
            new Scalarization()
                .setIsSuitableFor( call -> SuitabilityPredicate.BAD )
                .setDeviceExecution( (context, callback) -> ElemWiseUtil.forSubtractions(context.call(), callback) )
                .buildFunAlgorithm();

        setAlgorithm(
            Scalarization.class,
            scalarization.setImplementationFor(
                CPU.class,
                Scalarization.implementationForCPU()
                    .with(Fun.F64F64ToF64.triple(
                        ( a, b ) -> a - b,
                        ( a, b ) ->  1, // Deriving at input 0
                        ( a, b ) -> -1 // deriving input 1
                    ))
                    .with(Fun.F32F32ToF32.triple(
                        ( a, b ) -> a - b,
                        ( a, b ) ->  1, // Deriving at input 0
                        ( a, b ) -> -1 // deriving input 1
                    ))
                    .with(Fun.I32I32ToI32.triple(
                        ( a, b ) -> a - b,
                        ( a, b ) ->  1, // Deriving at input 0
                        ( a, b ) -> -1 // deriving input 1
                    ))
                    .get()
            )
            .setImplementationFor(
                OpenCLDevice.class,
                CLImplementation.compiler()
                        .arity( 3 )
                        .kernelSource( Scalarization.getKernelSource() )
                        .activationSource( "output = input1 - value;\n" )
                        .differentiationSource(
                            "if (d==0) {     \n" +//drn and src2 switch:
                            "    output = 1;  \n" +
                            "} else {         \n" +
                            "    output = -1;   " +
                            "}"
                        )
                        .kernelPostfix( this.getIdentifier() )
                        .execution(
                            call -> {
                                int offset = (call.input( Number.class, 2 ).isVirtual() || call.input( Number.class, 2 ).size() == 1)?1:0;
                                int gwz = call.input( Number.class, 0 ).size();
                                call.getDevice()
                                    .getKernel(call)
                                    .passAllOf(call.input( Number.class, 0 ))
                                    .passAllOf(call.input( Number.class, 0 ))
                                    .pass((float)call.input( Number.class, 1+offset).at(0).get().doubleValue())
                                    .pass( call.input( Number.class, 0 ).rank() )
                                    .pass( call.getValOf( Arg.DerivIdx.class ) )
                                    .call( gwz );
                            }
                        )
                        .build()
            )
        );

        //________________
        // BROADCASTING :

        Broadcast broadcast =
                (Broadcast)
                new Broadcast(ElemWiseUtil::forSubtractions)
                .setAutogradModeFor( call -> AutoDiffMode.BACKWARD_ONLY )
                .setSupplyADAgentFor(
                        ( Function f, ExecutionCall<? extends Device<?>> call ) ->
                        {
                            if ( call.autogradMode().allowsForward() )
                                throw new IllegalArgumentException("Broadcast implementation does not support forward-AD!");
                            Tsr<?> ctxDerivative = (Tsr<?>) call.getValOf(Arg.Derivative.class);
                            assert ctxDerivative == null;
                            int d = call.getDerivativeIndex();
                            Tsr<?> derivative = ElemWiseUtil.newTsrLike( call.input( d==0?1:0 ), 0 );
                            Tsr<?> toBeDerived = ElemWiseUtil.newTsrLike( call.input( d ), 0 );
                            Device device = call.getDevice();
                            return ADAgent.of( derivative )
                                        .withAD(
                                            target ->
                                                this.getAlgorithm( Broadcast.class )
                                                    .getImplementationFor( device )
                                                    .runAndGetFirstTensor(
                                                            ExecutionCall.of(
                                                                        toBeDerived.setIsVirtual(false),
                                                                        derivative,
                                                                        target.error()
                                                                    )
                                                                    .andArgs( Arg.DerivIdx.of(d) )
                                                                    .running( this )
                                                                    .on( device )
                                                    )
                                    );
                        }
                )
                .buildFunAlgorithm();

        setAlgorithm(
                Broadcast.class,
                broadcast
                    .setImplementationFor(
                        CPU.class,
                        Broadcast.implementationForCPU()
                                .with(Fun.F64F64ToF64.triple(
                                        ( a, b ) -> a - b,
                                        // In the context of broadcasting the traditional scalar derivative would be 1, broadcasting has different rules...
                                        ( a, b ) -> a + b, // Deriving at input 0
                                        ( a, b ) -> a - b // deriving input 1
                                ))
                                .with(Fun.F32F32ToF32.triple(
                                        ( a, b ) -> a - b,
                                        // In the context of broadcasting the traditional scalar derivative would be 1, broadcasting has different rules...
                                        ( a, b ) -> a + b, // Deriving at input 0
                                        ( a, b ) -> a - b // deriving input 1
                                ))
                                .get()
                    )
                    .setImplementationFor(
                            OpenCLDevice.class,
                            Broadcast.implementationForGPU( this.getIdentifier() )
                                    .with( "value += src1 - src2;\n" )
                                    .and( "value += src1 + src2 * -((d * 2) -1);\n" )
                    )
                );
    }


    @Contract(pure = true)
    @Override
    public String stringify( String[] children ) {
        StringBuilder reconstructed = new StringBuilder();
        for ( int i = 0; i < children.length; ++i ) {
            reconstructed.append( children[ i ] );
            if ( i < children.length - 1 ) {
                reconstructed.append(" - ");
            }
        }
        return "(" + reconstructed + ")";
    }

    @Override
    public String asDerivative( Function[] children, int derivationIndex) {
        return ( ( children[0].dependsOn(derivationIndex) ) ? "" : "-" ) +
                    Arrays.stream( children )
                    .filter( child -> child.dependsOn(derivationIndex) )
                    .map( child -> child.getDerivative(derivationIndex) )
                    .map( Object::toString )
                    .collect( Collectors.joining( " - " ) );
    }

    @Override
    public double calculate( double[] inputs, int j, int d, Function[] src ) {
        if ( j < 0 ) return calculate( inputs, d, src );
        if ( d < 0 ) {
            double result = src[ 0 ].call( inputs, j );
            for ( int i = 1; i < src.length; i++ ) {
                final double current = src[ i ].call( inputs, j );
                result -= current;
            }
            return result;
        } else {
            double derivative = 0;
            for ( int i = 0; i < src.length; i++ ) {
                if ( i == 0 )
                    derivative += src[ i ].derive( inputs, d, j );
                else
                    derivative -= src[ i ].derive( inputs, d, j );
            }
            return derivative;
        }
    }

    @Contract(pure = true)
    public static double calculate( double[] inputs, int d, Function[] src ) {
        if ( d < 0 ) {
            double result = src[ 0 ].call( inputs );
            for ( int i = 1; i < src.length; i++ ) {
                final double current = src[ i ].call( inputs );
                result -= current;
            }
            return result;
        } else {
            double derivative = 0;
            for ( int i = 0; i < src.length; i++ ) {
                if ( i == 0 )
                    derivative += src[ i ].derive( inputs, d );
                else
                    derivative -= src[ i ].derive( inputs, d );
            }
            return derivative;
        }
    }



}
