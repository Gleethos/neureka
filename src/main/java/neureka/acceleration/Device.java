package neureka.acceleration;

import neureka.Component;
import neureka.Neureka;
import neureka.Tsr;
import neureka.acceleration.host.HostCPU;
import neureka.acceleration.opencl.OpenCLDevice;
import neureka.acceleration.opencl.OpenCLPlatform;
import neureka.calculus.backend.ExecutionCall;
import neureka.calculus.frontend.assembly.FunctionParser;

import java.util.Arrays;
import java.util.Collection;

/**
 * This is the interface for implementations representing
 * devices capable of executing operations on tensors, namely the Tsr class.
 * Such instances are also components of tensors, which is why
 * the it extends the Component<Tsr> interface.
 */
public interface Device extends Component<Tsr>
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

        Device result = HostCPU.instance();
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
            Device first = OpenCLPlatform.PLATFORMS().get(0).getDevices().get(0);
            if( first!=null ) result = first;
        }
        return result;
    }

    void dispose();

    Device get( Tsr tensor );

    Device add( Tsr tensor );

    Device add( Tsr tensor, Tsr parent );

    boolean has( Tsr tensor );

    Device rmv( Tsr tensor );

    Device cleaning( Tsr tensor, Runnable action );

    Device overwrite64( Tsr tensor, double[] value );

    Device overwrite32( Tsr tensor, float[] value );

    Device swap( Tsr former, Tsr replacement );

    //Device execute( Tsr[] Tsrs, OperationType type, int d );
    Device execute( ExecutionCall call );

    double[] value64f( Tsr tensor );

    float[] value32f( Tsr tensor );

    double value64f( Tsr tensor, int index );

    float value32f( Tsr tensor, int index );

    Collection< Tsr > tensors();




}
