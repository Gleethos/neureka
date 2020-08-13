package neureka.acceleration;

import neureka.Component;
import neureka.Tsr;
import neureka.acceleration.host.HostCPU;
import neureka.acceleration.opencl.OpenCLPlatform;
import neureka.calculus.environment.ExecutionCall;
import neureka.calculus.environment.OperationType;

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
     * This method return OpenCLDevices matching
     * the given search parameter.
     * @param name The search parameter and name of the requested Device instance.
     * @return The found Device instance or simply the HostCPU instance by default.
     */
    static Device find(String name)
    {
        //TODO: Device plugin finding!
        Device[] result = {HostCPU.instance()};
        String search = name.toLowerCase();
        OpenCLPlatform.PLATFORMS().forEach( p ->
            p.getDevices().forEach( d ->{
                String str = (d.name()+" | "+d.vendor()+" | "+d.type()).toLowerCase();
                if(str.contains(search)) result[0] = d;
            }));
        if ( result[0]==HostCPU.instance() && name.equals("first") ) {
            Device first = OpenCLPlatform.PLATFORMS().get(0).getDevices().get(0);
            if( first!=null ) result[0] = first;
        }
        return result[0];
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
