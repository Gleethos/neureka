package neureka.function.factory.autograd;

public class PendingError {

    private int _toBeReceived;
    private GraphNode _location;

    public PendingError(GraphNode location, int toBeRecieved){
        _toBeReceived = toBeRecieved;
        _location = location;
    }



}
