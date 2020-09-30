package neureka.optimization;

import neureka.Component;
import neureka.Tsr;

public interface Optimizer<ValueType> extends Component<Tsr<ValueType>>
{

    void optimize(Tsr<ValueType> t);

}
