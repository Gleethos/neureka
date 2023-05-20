package neureka.backend.main.implementations.elementwise;

import neureka.Tensor;
import neureka.backend.api.ExecutionCall;
import neureka.backend.main.implementations.fun.api.ScalarFun;
import neureka.devices.host.CPU;
import neureka.ndim.NDimensional;

public class CPUElementwiseAssignFun extends CPUElementwiseFunction
{
    public CPUElementwiseAssignFun() { super(ScalarFun.IDENTITY); }


    @Override
    public Tensor<?> run(ExecutionCall<CPU> call )
    {
        assert call.arity() == 2;

        boolean allVirtual = call.validate().all( Tensor::isVirtual ).isValid();

        if ( allVirtual ) {
            call.input(Object.class, 0).mut().setDataAt(0, call.input(1).item() );
            assert call.input(0).isVirtual();
            assert call.input(1).isVirtual();
            return call.input(0);
        }

        call.input(0).mut().setIsVirtual(false);

        boolean isSimple = call.validate()
                                .allShare(Tensor::isVirtual)
                                .allShare(NDimensional::getNDConf)
                                .all( t -> t.getNDConf().isSimple() )
                                .isValid();
        if ( isSimple ) {
            Class<?> type = call.input(0).itemType();
            if ( type == Double.class ) {
                double[] output = call.input(0).mut().getDataForWriting(double[].class);
                double[] input = call.input(1).mut().getDataAs(double[].class);
                if ( input.length >= output.length ) {
                    System.arraycopy( input, 0, output, 0, call.input(0).size() );
                    return call.input(0);
                }
            } else if ( type == Integer.class ) {
                int[] output = call.input(0).mut().getDataForWriting(int[].class);
                int[] input = call.input(1).mut().getDataAs(int[].class);
                if ( input.length >= output.length ) {
                    System.arraycopy( input, 0, output, 0, call.input(0).size() );
                    return call.input(0);
                }
            } else if ( type == Float.class ) {
                float[] output = call.input(0).mut().getDataForWriting(float[].class);
                float[] input = call.input(1).mut().getDataAs(float[].class);
                if ( input.length >= output.length ) {
                    System.arraycopy( input, 0, output, 0, call.input(0).size() );
                    return call.input(0);
                }
            } else if ( type == Long.class ) {
                long[] output = call.input(0).mut().getDataForWriting(long[].class);
                long[] input = call.input(1).mut().getDataAs(long[].class);
                if ( input.length >= output.length ) {
                    System.arraycopy( input, 0, output, 0, call.input(0).size() );
                    return call.input(0);
                }
            } else if ( type == Boolean.class ) {
                boolean[] output = call.input(0).mut().getDataForWriting(boolean[].class);
                boolean[] input = call.input(1).mut().getDataAs(boolean[].class);
                if ( input.length >= output.length ) {
                    System.arraycopy( input, 0, output, 0, call.input(0).size() );
                    return call.input(0);
                }
            } else if ( type == Character.class ) {
                char[] output = call.input(0).mut().getDataForWriting(char[].class);
                char[] input = call.input(1).mut().getDataAs(char[].class);
                if ( input.length >= output.length ) {
                    System.arraycopy( input, 0, output, 0, call.input(0).size() );
                    return call.input(0);
                }
            } else if ( type == Byte.class ) {
                byte[] output = call.input(0).mut().getDataForWriting(byte[].class);
                byte[] input = call.input(1).mut().getDataAs(byte[].class);
                if ( input.length >= output.length ) {
                    System.arraycopy( input, 0, output, 0, call.input(0).size() );
                    return call.input(0);
                }
            } else if ( type == Short.class ) {
                short[] output = call.input(0).mut().getDataForWriting(short[].class);
                short[] input = call.input(1).mut().getDataAs(short[].class);
                if ( input.length >= output.length ) {
                    System.arraycopy( input, 0, output, 0, call.input(0).size() );
                    return call.input(0);
                }
            }
        }
        return super.run( call );
    }


}
