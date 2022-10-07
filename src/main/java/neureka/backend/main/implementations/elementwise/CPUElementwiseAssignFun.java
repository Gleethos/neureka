package neureka.backend.main.implementations.elementwise;

import neureka.Tsr;
import neureka.backend.api.ExecutionCall;
import neureka.backend.main.implementations.fun.api.ScalarFun;
import neureka.devices.host.CPU;
import neureka.ndim.NDimensional;

public class CPUElementwiseAssignFun extends CPUElementwiseFunction
{
    public CPUElementwiseAssignFun() { super(ScalarFun.IDENTITY); }


    @Override
    public Tsr<?> run(ExecutionCall<CPU> call)
    {
        boolean isSimple = call.validate()
                                .allShare(Tsr::isVirtual)
                                .allShare(NDimensional::getNDConf)
                                .isValid();
        if ( isSimple ) {
            Class<?> type = call.input(0).itemType();
            if ( type == Double.class ) {
                double[] output = call.input(0).getMut().getDataForWriting(double[].class);
                double[] input = call.input(1).getMut().getDataAs(double[].class);
                if ( input.length == output.length ) {
                    System.arraycopy(input, 0, output, 0, input.length);
                    return call.input(0);
                }
            } else if ( type == Integer.class ) {
                int[] output = call.input(0).getMut().getDataForWriting(int[].class);
                int[] input = call.input(1).getMut().getDataAs(int[].class);
                if ( input.length == output.length ) {
                    System.arraycopy(input, 0, output, 0, input.length);
                    return call.input(0);
                }
            } else if ( type == Float.class ) {
                float[] output = call.input(0).getMut().getDataForWriting(float[].class);
                float[] input = call.input(1).getMut().getDataAs(float[].class);
                if ( input.length == output.length ) {
                    System.arraycopy(input, 0, output, 0, input.length);
                    return call.input(0);
                }
            } else if ( type == Long.class ) {
                long[] output = call.input(0).getMut().getDataForWriting(long[].class);
                long[] input = call.input(1).getMut().getDataAs(long[].class);
                if ( input.length == output.length ) {
                    System.arraycopy(input, 0, output, 0, input.length);
                    return call.input(0);
                }
            } else if ( type == Boolean.class ) {
                boolean[] output = call.input(0).getMut().getDataForWriting(boolean[].class);
                boolean[] input = call.input(1).getMut().getDataAs(boolean[].class);
                if ( input.length == output.length ) {
                    System.arraycopy(input, 0, output, 0, input.length);
                    return call.input(0);
                }
            } else if ( type == Character.class ) {
                char[] output = call.input(0).getMut().getDataForWriting(char[].class);
                char[] input = call.input(1).getMut().getDataAs(char[].class);
                if ( input.length == output.length ) {
                    System.arraycopy(input, 0, output, 0, input.length);
                    return call.input(0);
                }
            }
        }
        return super.run( call );
    }


}
