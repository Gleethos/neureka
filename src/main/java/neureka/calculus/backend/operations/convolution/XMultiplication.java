package neureka.calculus.backend.operations.convolution;

import neureka.Tsr;
import neureka.acceleration.Device;
import neureka.acceleration.host.execution.HostExecutor;
import neureka.acceleration.opencl.execution.CLExecutor;
import neureka.autograd.ADAgent;
import neureka.calculus.Function;
import neureka.calculus.backend.operations.AbstractOperationType;
import neureka.calculus.backend.ExecutionCall;
import neureka.calculus.backend.operations.OperationType;
import neureka.calculus.backend.implementations.OperationTypeImplementation;
import neureka.calculus.backend.implementations.Convolution;
import neureka.calculus.frontend.assembly.FunctionBuilder;

import java.util.List;

public class XMultiplication extends AbstractOperationType
{

    public XMultiplication(){

        super(
                "multiply",
                "x",
                2,
                true,
                false,
                false,
                false
        );

        setStringifier(
                children -> {
                    StringBuilder reconstructed = new StringBuilder();
                    for ( int i = 0; i < children.size(); ++i ) {
                        reconstructed.append( children.get(i) );
                        if ( i < children.size() - 1 ) {
                            reconstructed.append(" x ");
                        }
                    }
                    return "(" + reconstructed + ")";
                }
        );

        OperationTypeImplementation.RecursiveJunctionAgent rja = (call, goDeeperWith)->
        {
            Tsr[] tsrs = call.getTensors();
            Device device = call.getDevice();
            int d = call.getDerivativeIndex();
            OperationType type = call.getType();

            Tsr alternative = null;
            if (tsrs.length > 3) {
                if (d < 0) {
                    Tsr[] reduction = new Tsr[]{tsrs[0], tsrs[1], tsrs[2]};
                    alternative = goDeeperWith.apply(
                            new ExecutionCall<>(device, reduction, d, type)
                    );
                    tsrs[0] = reduction[0];

                    reduction = Utility._offsetted(tsrs, 1);
                    alternative = goDeeperWith.apply(
                            new ExecutionCall<>(device, reduction, d, type)
                    );
                    tsrs[0] = reduction[0];
                }
                return alternative;
            } else {
                if ( call.getType().getOperator().equals("x") ) {
                    if (d >= 0) {
                        if (d == 0) tsrs[0] = tsrs[2];
                        else tsrs[0] = tsrs[1];
                        return tsrs[0];
                    } else {
                        call.mutateArguments( t -> new Tsr[]{t[0], t[1], t[2]} );
                    }
                } else if ( call.getType().getOperator().equals("x"+ ((char) 187)) ) {
                    call.mutateArguments( t -> new Tsr[]{t[2], t[1], t[0]} );
                }
                return alternative;
            }
        };

        DefaultOperatorCreator<TertiaryNDXConsumer> convolutionCreator =
                (inputs, d) -> {
                    double[] t1_val = inputs[1].value64();
                    double[] t2_val = inputs[2].value64();
                    if (d < 0) {
                        return (t0Idx, t1Idx, t2Idx) -> t1_val[inputs[1].i_of_idx(t1Idx)] * t2_val[inputs[2].i_of_idx(t2Idx)];
                    } else {
                        return (t0Idx, t1Idx, t2Idx) -> {
                            if (d == 0) return t2_val[inputs[2].i_of_idx(t2Idx)];
                            else return t1_val[inputs[1].i_of_idx(t1Idx)];
                        };
                    }
                };

        Convolution convolution = new Convolution()
                .setADAnalyzer(
                    call -> {
                        if ( call.getType().supports(Convolution.class) ) return false;
                        if ( call.getType().getOperator().equals(",") ) return false; //Reshape
                        Tsr last = null;
                        for ( Tsr t : call.getTensors() ) {
                            if ( last != null && !last.shape().equals(t.shape()) ) return false;
                            last = t; // Note: shapes are cached!
                        }
                        return true;
                    }
        ).setADAgentSupplier(
            ( Function f, ExecutionCall<Device> call, boolean forward ) ->
            {
                Tsr ctxDerivative = (Tsr)call.getAt("derivative");
                if ( forward ) throw new IllegalArgumentException("Convolution of does not support forward-AD!");

                Function mul = Function.Detached.MUL;
                Tsr[] inputs = call.getTensors();
                int d = call.getDerivativeIndex();

                Function invX = FunctionBuilder.build(
                        "I[0]" + getOperator() + ">>I[1]" + getOperator() + ">>I[2]",
                        false
                );
                Tsr deriv = f.derive(inputs, d);
                return new ADAgent(
                        deriv
                ).withForward(
                        (node, forwardDerivative) -> mul.call(new Tsr[]{forwardDerivative, deriv})
                ).withBackward(
                        (t, error) -> invX.call(new Tsr[]{error, deriv, new Tsr(t.getPayload().shape(), 0)})
                );
            }
        ).setCallHock(
                (caller, call) -> {
                    if ( !caller.isFlat() ) return null;
                    if ( call.getType().getOperator().equals("x") ) {

                        Tsr[] inputs = call.getTensors();
                        Tsr[] tsrs = new Tsr[]{null, inputs[0], inputs[1]};// _src_acti(inputs, j, -1, 1);
                        tsrs[0] = (call.getDerivativeIndex() < 0)
                                ? new Tsr(Tsr.Utility.Indexing.shpOfCon(tsrs[1].getNDConf().shape(), tsrs[2].getNDConf().shape()))
                                : null;

                        for (Tsr t : tsrs) if (t != null) t.setIsVirtual(false);
                        call.getDevice().execute(call.withNew(tsrs));
                        return tsrs[0];
                    } else {
                        if (call.getDerivativeIndex() < 0) {
                            Tsr[] tsrs = caller.srcActivation(call.getTensors(), call.getJ(), -1, 0);
                            for ( Tsr t : tsrs ) t.setIsVirtual(false);
                            call.getDevice().execute( new ExecutionCall( call.getDevice(), tsrs, 0, call.getType() ) );
                            if ( call.getType().getId() == OperationType.instance("x>>").getId()) return tsrs[2];
                            else return tsrs[0];
                        }
                    }
                    return null;
                }
        ).setRJAgent(
               rja
        ).setDrainInstantiation(
                call -> {
                    Tsr[] tsrs = call.getTensors();
                    Device device = call.getDevice();
                    if ( tsrs[0] == null ) // Creating a new tensor:
                    {
                        int[] shp = Tsr.Utility.Indexing.shpOfCon(tsrs[1].getNDConf().shape(), tsrs[2].getNDConf().shape());
                        Tsr output = new Tsr( shp, 0.0 );
                        output.setIsVirtual(false);
                        device.add(output);
                        tsrs[0] = output;
                    }
                    return call;
                }
        );

        setImplementation(
                Convolution.class,
                convolution
                        .setExecutor(
                                HostExecutor.class,
                                new HostExecutor(
                                        call ->
                                                call.getDevice().getExecutor()
                                                        .threaded (
                                                                call.getTensor(0).size(),
                                                                ( start, end ) ->
                                                                        Convolution.convolve (
                                                                                call.getTensor(0), call.getTensor(1), call.getTensor(2),
                                                                                call.getDerivativeIndex(), start, end,
                                                                                convolutionCreator.create(
                                                                                        call.getTensors(),
                                                                                        -1//call.getDerivativeIndex()
                                                                                )
                                                                        )
                                                        ),
                                        3
                                )
                        ).setExecutor(
                        CLExecutor.class,
                        new CLExecutor(
                                call -> {
                                    int offset = (call.getTensor(0) != null) ? 0 : 1;
                                    int gwz = (call.getTensor(0) != null) ? call.getTensor(0).size() : call.getTensor(1).size();
                                    call.getDevice().getKernel(call)
                                            .pass(call.getTensor(offset))
                                            .pass(call.getTensor(offset + 1))
                                            .pass(call.getTensor(offset + 2))
                                            .pass(call.getTensor(0).rank())
                                            .pass(call.getDerivativeIndex())//call.getDerivativeIndex()
                                            .call(gwz);
                                },
                                3,
                                convolution.getKernelSource(), // kernelSource
                                "value = src1 * src2;\n",
                                "value += handle * drain;\n",
                                this // OperationType
                        )
                )
        );
        new AbstractOperationType(
                "inv_convolve_mul_left", ((char) 171) + "x",
                3,
                true,
                false,
                false,
                false
        ){

            @Override
            public double calculate(double[] inputs, int j, int d, List<Function> src) {
            return src.get(0).call( inputs, j );
            }
        }
                .setImplementation(Convolution.class, convolution)
                .setStringifier(
                    children -> {
                        StringBuilder reconstructed = new StringBuilder();
                        for ( int i = 0; i < children.size(); ++i ) {
                            reconstructed.append( children.get(i) );
                            if ( i < children.size() - 1 ) {
                                reconstructed.append(" "+((char) 171) + "x ");
                            }
                        }
                        return "(" + reconstructed + ")";
                    }
                );

        new AbstractOperationType(
                "inv_convolve_mul_right", "x" + ((char) 187),
                3,
                true,
                false,
                false,
                false
        ){
            @Override
            public double calculate(double[] inputs, int j, int d, List<Function> src){
                return 0;
            }
        }.setImplementation(Convolution.class, convolution)
                .setStringifier(
                        children -> {
                            StringBuilder reconstructed = new StringBuilder();
                            for ( int i = 0; i < children.size(); ++i ) {
                                reconstructed.append( children.get(i) );
                                if ( i < children.size() - 1 ) {
                                    reconstructed.append(" x" + ((char) 187)+" ");
                                }
                            }
                            return "(" + reconstructed + ")";
                        }
                );




    }


    @Override
    public double calculate(double[] inputs, int j, int d, List<Function> src) {
            return src.get(0).call( inputs, j );
    }
}
