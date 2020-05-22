package neureka.optimization;

import neureka.Component;
import neureka.Tsr;

public interface Optimizer extends Component<Tsr>
{

    void optimize(Tsr t);

}
