package neureka.framing.states;

public interface At<KeyType, ReturnType> {

    ReturnType at( KeyType key );

}
