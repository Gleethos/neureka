package neureka.backend.main.operations.operator;

import neureka.Neureka;
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
import neureka.backend.api.template.algorithms.AbstractDeviceAlgorithm;
import neureka.devices.Device;
import neureka.devices.host.CPU;
import neureka.devices.opencl.OpenCLDevice;
import neureka.ndim.NDimensional;
import org.jetbrains.annotations.Contract;

public class Modulo extends AbstractOperation {

    public Modulo()
    {
        super(
                new OperationBuilder()
                        .identifier(       "modulo"    )
                        .operator(         "%"         )
                        .arity(            -1          )
                        .isOperator(       true        )
                        .isIndexer(        false       )
                        .isDifferentiable( true        )
                        .isInline(         false       )
        );

        //_____________________
        // DEFAULT OPERATION :

        Operator operator =
                new Operator(ElemWiseUtil::forDivisionsOrModuli)
                       .setSupplyADAgentFor( getDefaultAlgorithm() )
                       .buildFunAlgorithm();

        setAlgorithm(
            Operator.class,
            operator.setImplementationFor(
                CPU.class,
                Operator.implementationForCPU()
                    .with(Fun.F64F64ToF64.triple(
                        ( a, b ) -> a % b,
                        ( a, b ) -> 1 / b, // Deriving at input 0
                        ( a, b ) -> -(a / Math.pow(b, 2)) // deriving input 1
                    ))
                    .with(Fun.F32F32ToF32.triple(
                        ( a, b ) -> a % b,
                        ( a, b ) -> 1f / b, // Deriving at input 0
                        ( a, b ) -> (float) -(a / Math.pow(b, 2)) // deriving input 1
                    ))
                    .with(Fun.I32I32ToI32.triple(
                        ( a, b ) -> a % b,
                        ( a, b ) -> 1 / b, // Deriving at input 0
                        ( a, b ) -> (int) Math.round(-a / Math.pow(b, 2)) // deriving input 1
                    ))
                    .with(Fun.I64I64ToI64.triple(
                        ( a, b ) -> a % b,
                        ( a, b ) -> 1 / b, // Deriving at input 0
                        ( a, b ) -> Math.round(-a / Math.pow(b, 2)) // deriving input 1
                    ))
                    .get()
            )
            .setImplementationFor(
                OpenCLDevice.class,
                Operator.implementationForGPU( this.getIdentifier() )
                        .with( "output = ((int)input1) % ((int)input2);\n" )
                        .and(
                                "if ( d==0 ) {                                        \n" +
                                        "    output = 1/input2;                               \n" +
                                        "} else {                                             \n" +
                                        "    output = -input2 / (float) pow(input1, 2.0f);    \n" +
                                        "}"
                        )
            )
        );



        //________________
        // BROADCASTING :

        Broadcast broadcast = new Broadcast( AbstractDeviceAlgorithm::executeDeviceAlgorithm )
            .setAutogradModeFor(
                    call -> call
                                .validate().allNotNullHaveSame(NDimensional::shape)
                                .ifValid(AutoDiffMode.FORWARD_AND_BACKWARD)
                                .orElse(AutoDiffMode.BACKWARD_ONLY)
            )
            .setSupplyADAgentFor(
                ( Function f, ExecutionCall<? extends Device<?>> call ) ->
                {
                    if ( call.autogradMode().allowsForward() )
                        throw new IllegalArgumentException("Broadcast implementation does not support forward-AD!");
                    Tsr<?> ctxDerivative = (Tsr<?>) call.getValOf(Arg.Derivative.class);
                    Function mul = Neureka.get().backend().getFunction().mul();
                    if ( ctxDerivative != null ) {
                        return ADAgent.of( ctxDerivative )
                                        .withAD( target -> mul.execute( target.error(), ctxDerivative ) );
                    }
                    int d = call.getDerivativeIndex();
                    Tsr<?> derivative = f.executeDerive( call.inputs(), d );
                    return ADAgent.of( derivative )
                                    .withAD( target -> mul.execute( target.error(), derivative ) );
                }
            )
            .buildFunAlgorithm();

        setAlgorithm(
            Broadcast.class,
            broadcast.setImplementationFor(
                CPU.class,
                Broadcast.implementationForCPU()
                        .with(Fun.F64F64ToF64.triple(
                            ( a, b ) -> a % b,
                            ( a, b ) -> 1 / b, // Deriving at input 0
                            ( a, b ) -> -(a / Math.pow(b, 2)) // deriving input 1
                        ))
                        .with(Fun.F32F32ToF32.triple(
                            ( a, b ) -> a % b,
                            ( a, b ) -> 1 / b, // Deriving at input 0
                            ( a, b ) -> (float) -(a / Math.pow(b, 2)) // deriving input 1
                        ))
                        .get()
            )
            .setImplementationFor(
                OpenCLDevice.class,
                Broadcast.implementationForGPU( this.getIdentifier() )
                        .with( "value = ((int)src1) % ((int)src2);\n" )
                        .and(
                            "if ( d == 0 ) {\n" +
                            "    value += (1/handle) * drain;\n" +//TODO: this is probably wrong!
                            "} else {\n" +
                            "    value += (-(handle /(float)pow(target, (float)2)) ) * drain;\n" +
                            "}"
                        )
            )
        );

        //___________________________
        // TENSOR SCALAR OPERATION :

        Scalarization scalarization = new Scalarization()
                .setIsSuitableFor( call -> SuitabilityPredicate.BAD )
                .setAutogradModeFor(
                        call -> call
                                .validate().allNotNullHaveSame(NDimensional::shape)
                                .ifValid(AutoDiffMode.FORWARD_AND_BACKWARD)
                                .orElse(AutoDiffMode.BACKWARD_ONLY)
                )
                .setDeviceExecution( (call, callback) -> AbstractDeviceAlgorithm.executeDeviceAlgorithm( call, callback ) )
                .buildFunAlgorithm();

        setAlgorithm(
            Scalarization.class,
            scalarization.setImplementationFor(
                CPU.class,
                Scalarization.implementationForCPU()
                    .with(Fun.F64F64ToF64.triple(
                        ( a, b ) -> a % b,
                        ( a, b ) -> 1 / b, // Deriving at input 0
                        ( a, b ) -> -(a / Math.pow(b, 2)) // deriving input 1
                    ))
                    .with(Fun.F32F32ToF32.triple(
                        ( a, b ) -> a % b,
                        ( a, b ) -> 1 / b, // Deriving at input 0
                        ( a, b ) -> (float) -(a / Math.pow(b, 2)) // deriving input 1
                    ))
                    .with(Fun.I32I32ToI32.triple(
                        ( a, b ) -> a % b,
                        ( a, b ) -> (int) Math.round(1d / b), // Deriving at input 0
                        ( a, b ) -> (int) Math.round(-a / Math.pow(b, 2)) // deriving input 1
                    ))
                    .with(Fun.I64I64ToI64.triple(
                        ( a, b ) -> a % b,
                        ( a, b ) -> Math.round(1d / b), // Deriving at input 0
                        ( a, b ) -> Math.round(-a / Math.pow(b, 2)) // deriving input 1
                    ))
                    .get()
            )
            .setImplementationFor(
                OpenCLDevice.class,
                CLImplementation
                    .compiler()
                    .arity( 3 )
                    .kernelSource( Scalarization.getKernelSource() )
                    .activationSource( "output = ((int)input1) % ((int)value);     \n" )
                    .differentiationSource(
                        "if ( d == 0 ) {                               \n" +
                        "    output = 1/value;                           \n" +
                        "} else {                                        \n" +
                        "    output = -value /(float)pow(input1, 2.0f);  \n" +
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
                                .pass( call.input( Number.class, 1 + offset ).at( 0 ).get().floatValue() )
                                .pass( call.input( Number.class, 0 ).rank() )
                                .pass( call.getValOf( Arg.DerivIdx.class ) )
                                .call( gwz );

                            return call.input( 0 );
                        }
                    )
                    .build()
            )
        );

    }

    @Contract(pure = true)
    public static double calculate( double[] inputs, int d, Function[] src ) {
        if ( d < 0 ) {
            double result = src[ 0 ].call( inputs );
            for ( int i = 1; i < src.length; i++ ) {
                final double current = src[ i ].call( inputs );
                result %= current;
            }
            return result;
        }
        else return src[ 0 ].derive( inputs, d );
    }

    @Override
    public String asDerivative( Function[] children, int derivationIndex) {
        return children[ 0 ].getDerivative(derivationIndex).toString();
    }

    @Override
    public double calculate( double[] inputs, int j, int d, Function[] src ) {
        if ( j < 0 ) return calculate( inputs, d, src );
        if ( d < 0 ) {
            double result = src[ 0 ].call( inputs, j );
            for ( int i = 1; i < src.length; i++ ) {
                final double current = src[ i ].call( inputs, j );
                result %= current;
            }
            return result;
        }
        else
            return src[ 0 ].derive( inputs, d, j );
    }





}
