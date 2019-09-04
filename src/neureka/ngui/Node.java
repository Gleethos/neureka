package neureka.ngui;

import neureka.core.T;
import neureka.core.function.IFunction;

import java.util.ArrayList;

public class Node {

    private ArrayList<Node> input = new ArrayList<>();

    private IFunction function;

    private T tensor;

    public T getTensor(){
        return tensor;
    }
    public void setTensor(T t){
        this.tensor = t;
    }

    public ArrayList<Node> getConnection(){
        return input;
    }

    public void connect(Node n){
        if(!input.contains(n)){
            input.add(n);
        }
    }
    public void disconnect(Node n){
        input.remove(n);
    }


}
