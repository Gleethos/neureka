package neureka.autograd;

import java.util.Collections;
import java.util.List;

public class BackPropBridge<V>
{
    private final GraphNode<V> _target;
    private final List<ADAgent> _agents;

    public BackPropBridge(GraphNode<V> target, List<ADAgent> agents ) {
        _target = target;
        _agents = Collections.unmodifiableList(agents);
    }

    public GraphNode<V> target() { return _target; }

    public List<ADAgent> agents() { return _agents; }
}
