package neureka.ngui.touch.canvas;

import javafx.scene.canvas.Canvas;

import java.util.ArrayList;

public class Layer extends Canvas {

    private ArrayList<NPanelRepaintSpace> RepaintQueue = null;
    private int order;


    Layer(int order){
        super();
        this.order = order;
    }


    public ArrayList<NPanelRepaintSpace> getRepaintQueue(){
        return RepaintQueue;
    }
    public void setRepaintQueue(ArrayList<NPanelRepaintSpace> queue){
        this.RepaintQueue = queue;
    }

    public int getOrder(){
        return this.order;
    }

}
