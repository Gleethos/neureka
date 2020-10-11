package neureka.device;

import neureka.Component;
import neureka.Neureka;
import neureka.Tsr;
import neureka.device.host.HostCPU;
import neureka.device.opencl.OpenCLDevice;
import neureka.device.opencl.OpenCLPlatform;
import neureka.calculus.backend.ExecutionCall;
import neureka.calculus.frontend.assembly.FunctionParser;

import java.util.Arrays;
import java.util.Collection;

/**
 * This is the interface for implementations representing
 * devices capable of executing operations on tensors, namely the Tsr<ValueType> class.
 * Such instances are also components of tensors, which is why
 * this interface extends the Component &lt; Tsr<ValueType> &gt; interface.
 */
public interface Device<ValueType> extends Component<Tsr<ValueType>>
{
    /**
     * This method return Device instances matching
     * the given search parameter.
     * @param name The search parameter and name of the requested Device instance.
     * @return The found Device instance or simply the HostCPU instance by default.
     */
    static Device find(String name)
    {
        String search = name.toLowerCase();
        boolean probablyWantsGPU = Arrays.stream(
                new String[]{
                        "gpu", "nvidia", "amd", "intel", "opencl", "fpga"
                }
        ).anyMatch(search::contains);

        if ( !Neureka.instance().canAccessOpenCL() ) {
            if ( probablyWantsGPU ) {
                return null; // User wants OpenCL but cannot have it :/
            } else return HostCPU.instance();
        }

        Device<Number> result = HostCPU.instance();
        double score = FunctionParser.similarity( "jvm native host cpu threaded", search );
        if ( probablyWantsGPU ) score /= 10; // HostCPU instance is most likely not meant!

        for ( OpenCLPlatform p : OpenCLPlatform.PLATFORMS() ) {
            for ( OpenCLDevice d : p.getDevices() ) {
                String str = ("opencl | "+d.type()+" | "+d.name()+" | "+d.vendor()).toLowerCase();
                double similarity = FunctionParser.similarity( str, search );
                if ( similarity > score || str.contains(search) ) {
                    result = d;
                    score = similarity;
                }
            }
        }
        if ( result == HostCPU.instance() && name.equals("first") ) {
            Device<Number> first = OpenCLPlatform.PLATFORMS().get(0).getDevices().get(0);
            if( first!=null ) result = first;
        }
        return result;
    }

    void dispose();

    Device get( Tsr<ValueType> tensor );

    Device add( Tsr<ValueType> tensor );

    Device add( Tsr<ValueType> tensor, Tsr<ValueType> parent );

    boolean has( Tsr<ValueType> tensor );

    Device rmv( Tsr<ValueType> tensor );

    Device cleaning( Tsr<ValueType> tensor, Runnable action );

    Device overwrite64( Tsr<ValueType> tensor, double[] value );

    Device overwrite32( Tsr<ValueType> tensor, float[] value );

    Device swap( Tsr<ValueType> former, Tsr<ValueType> replacement );

    Device execute( ExecutionCall call );

    double[] value64f( Tsr<ValueType> tensor );

    float[] value32f( Tsr<ValueType> tensor );

    double value64f( Tsr<ValueType> tensor, int index );

    float value32f( Tsr<ValueType> tensor, int index );

    Collection< Tsr<ValueType> > tensors();




}
