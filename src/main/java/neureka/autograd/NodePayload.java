package neureka.autograd;

import neureka.Tsr;
import neureka.dtype.DataType;

import java.lang.ref.WeakReference;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

final class NodePayload<V> {

    private final int _payloadReferenceVersion;

    private final int[] _payloadShape;

    private final DataType<V> _payloadDataType;

    private final WeakReference<Tsr<V>> _payload;


    public NodePayload( Tsr<V> p ) {
        if ( p == null ) {
            _payload = null;
            _payloadShape = null;
            _payloadReferenceVersion = -1;
            _payloadDataType = null;
        }
        else {
            assert !p.isUndefined();
            _payload = new WeakReference<>( p );
            _payloadShape = p.getNDConf().shape();
            _payloadReferenceVersion = p.getVersion();
            _payloadDataType = p.getDataType();
        }
    }

    public DataType<V> payloadDataType() { return _payloadDataType; }

    public int payloadReferenceVersion() { return _payloadReferenceVersion; }

    /**
     *  The value of a graph node is the tensor to which it belongs (is a component of).  <br><br>
     *  Meaning it is the tensor owning this {@link GraphNode} component.
     *  It is referenced weakly because it might not be needed any more (Not referenced inside AD-Agent for example)
     *  and can therefore be garbage collected.
     *
     *  Warning: This method might return null because
     *           the payload is weakly referenced!
     *           Meaning that it might get garbage collected.
     *
     * @return The tensor payload of this graph-node.
     */
    public Tsr<V> getPayload() { return ( _payload == null ? null : _payload.get() ); }

    /**
     *  Note: This method will never return null even if the actual payload tensor was garbage collected.
     *  This is because the {@link GraphNode} will remember the shape of the tensor.
     *
     *  @return The shape of the payload tensor represented by this {@link GraphNode}.
     */
    public List<Integer> getPayloadShape() {
        return _payloadShape == null ? null : Arrays.stream(_payloadShape).boxed().collect(Collectors.toList());
    }

}
