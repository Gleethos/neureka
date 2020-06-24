package neureka.calculus.environment.executors;

import neureka.Tsr;
import neureka.calculus.environment.Type;

public class Broadcast extends AbstractTypeExecutor<Type.OperatorCreator>
{
    public Broadcast(String strBroadcast, String strDeriviation, Type.OperatorCreator creator){
        super(strBroadcast, strDeriviation, creator);
    }

    @Override
    public boolean canHandle(ExecutionCall call)
    {
        int maxRank = 0;
        for ( Tsr t : call.getTensors() ) if( t!=null && t.rank()>maxRank) maxRank = t.rank();

        for ( int i = 0; i < maxRank; i++ )
        {
            int currentDim = -1;
            for( Tsr t : call.getTensors() )
            {
                if( t!=null && i<t.rank() ) {
                    if ( currentDim == -1 ) currentDim = t.shape(i);
                    else if ( currentDim!=t.shape(i) && currentDim!=1 && t.shape(i)!=1 ) return false;
                }
            }
        }
        return true;
    }
}
