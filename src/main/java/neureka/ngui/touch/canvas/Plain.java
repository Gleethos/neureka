package neureka.ngui.touch.canvas;

import javafx.event.EventHandler;
import javafx.scene.Group;
import javafx.scene.canvas.Canvas;
import javafx.scene.canvas.GraphicsContext;
import javafx.scene.control.Button;
import javafx.scene.input.MouseEvent;
import javafx.scene.input.ScrollEvent;
import javafx.scene.layout.Pane;
import javafx.scene.paint.Color;
import javafx.scene.shape.ArcType;
import javafx.scene.transform.Affine;
import neureka.ngui.touch.canvas.AnimatedZoomOperator;
import neureka.ngui.touch.canvas.NMap;

import java.util.Random;

public class Plain extends Pane {

    NMap map = new NMap(0, 0, 1000);
    Plain(){

        this.setOnMouseEntered(new EventHandler<MouseEvent>() {
            public void handle(MouseEvent me) {
                System.out.println("Mouse entered");
            }
        });
        this.setOnMouseExited(new EventHandler<MouseEvent>() {
            public void handle(MouseEvent me) {
                System.out.println("Mouse exited");
            }
        });
        this.setOnMousePressed(new EventHandler<MouseEvent>() {
            public void handle(MouseEvent me) {
                System.out.println("Mouse pressed");
                orgSceneX = me.getX();
                orgSceneY = me.getY();
            }
        });
        this.setOnMouseDragged(new EventHandler<MouseEvent>() {
            public void handle(MouseEvent me) {
                System.out.println("Mouse dragged");
                double offsetX = me.getX()-orgSceneX;
                double offsetY = me.getY()-orgSceneY;
                orgSceneX = me.getX();
                orgSceneY = me.getY();
                tX+=offsetX;
                tY+=offsetY;
                setTranslateX(me.getX() + getTranslateX() - 120);
                setTranslateY(me.getY() + getTranslateY() - 50);

            }
        });

        // Create operator
        AnimatedZoomOperator zoomOperator = new AnimatedZoomOperator();

        // Listen to scroll events (similarly you could listen to a button click, slider, ...)
        Pane c = this;
        setOnScroll(new EventHandler<ScrollEvent>() {
            @Override
            public void handle(ScrollEvent event) {
                double zoomFactor = 1.5;
                if (event.getDeltaY() <= 0) {
                    // zoom out
                    zoomFactor = 1 / zoomFactor;
                }
                oldScale = scale;
                scale *= zoomFactor;
                //zoomOperator.zoom(c, zoomFactor, event.getSceneX(), event.getSceneY());
                sX = event.getSceneX();
                sY = event.getSceneY();
            }
        });

    }
    private double orgSceneX, orgSceneY;
    private double sX, sY;

    private  double tX, tY;
    private  double scale = 1;
    private double oldScale = 1;

    public void update(){

        this.getChildren().forEach((c)->{
            if(c instanceof Canvas){
                this.paint(((Canvas)c).getGraphicsContext2D());
            }
        });



    }

    boolean test = true;

    private void paint(GraphicsContext gc){
        if(test){
            //gc.translate(5, 5);test = false;
        }
        Affine tf = gc.getTransform();//

        //tf.setMxx(scale);
        //tf.setMyy(scale);
        //tf.setTx(tX);
        //tf.setTy(tY);


        gc.setTransform(tf);

        System.out.println("update");
        gc.clearRect(0, 0, 300, 300);
        gc.setFill(Color.GREEN);
        gc.setStroke(Color.BLUE);
        gc.setLineWidth(5+(new Random()).nextInt()%10);
        gc.strokeLine(40, 10, 10, 40);
        gc.fillOval(10, 60, 30, 30);
        gc.strokeOval(60, 60, 30, 30);
        gc.fillRoundRect(110, 60, 30, 30, 10, 10);
        gc.strokeRoundRect(160, 60, 30, 30, 10, 10);
        gc.fillArc(10, 110, 30, 30, 45, 240, ArcType.OPEN);
        gc.fillArc(60, 110, 30, 30, 45, 240, ArcType.CHORD);
        gc.fillArc(110, 110, 30, 30, 45, 240, ArcType.ROUND);
        gc.strokeArc(10, 160, 30, 30, 45, 240, ArcType.OPEN);
        gc.strokeArc(60, 160, 30, 30, 45, 240, ArcType.CHORD);
        gc.strokeArc(110, 160, 30, 30, 45, 240, ArcType.ROUND);
        gc.strokeRect(-20, 30, 30, 30);
        gc.fillPolygon(new double[]{10, 40, 10, 40},
                new double[]{210, 210, 240, 240}, 4);
        gc.strokePolygon(new double[]{60, 90, 60, 90},
                new double[]{210, 210, 240, 240}, 4);
        gc.strokePolyline(new double[]{110, 140, 110, 140},
                new double[]{210, 210, 240, 240}, 4);
    }




}
