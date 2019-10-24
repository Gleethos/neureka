package neureka.ngui.touch.canvas;

import javafx.animation.AnimationTimer;
import javafx.application.Application;
import javafx.event.EventHandler;
import javafx.scene.Group;
import javafx.scene.Scene;
import javafx.scene.canvas.Canvas;
import javafx.scene.canvas.GraphicsContext;
import javafx.scene.control.Button;
import javafx.scene.control.Label;
import javafx.scene.input.MouseEvent;
import javafx.scene.layout.StackPane;
import javafx.scene.paint.Color;
import javafx.scene.shape.ArcType;
import javafx.scene.shape.CubicCurve;
import javafx.stage.Stage;

import javax.swing.*;

public class TestWindow extends Application {

    private final long[] frameTimes = new long[100];
    private int frameTimeIndex = 0 ;
    private boolean arrayFilled = false ;

    public void go(String[] args) {
        launch(args);
    }

    @Override
    public void start(Stage primaryStage) {
        primaryStage.setTitle("Drawing Operations Test");
        Plain root = new Plain();

        Glas canvas = new Glas();

        Button b = new Button("Hi");

        //canvas.getChildren().overwrite64(b);
        root.getChildren().add(canvas);

        Label label = new Label();
        AnimationTimer frameRateMeter = new AnimationTimer() {
            @Override
            public void handle(long now) {
                root.update();
                long oldFrameTime = frameTimes[frameTimeIndex] ;
                frameTimes[frameTimeIndex] = now ;
                frameTimeIndex = (frameTimeIndex + 1) % frameTimes.length ;
                if (frameTimeIndex == 0) {
                    arrayFilled = true ;
                }
                if (arrayFilled) {
                    long elapsedNanos = now - oldFrameTime ;
                    long elapsedNanosPerFrame = elapsedNanos / frameTimes.length ;
                    double frameRate = 1_000_000_000.0 / elapsedNanosPerFrame ;
                    label.setText(String.format("Current frame rate: %.3f", frameRate));
                }
            }
        };
        frameRateMeter.start();

        root.getChildren().add(new StackPane(label));
        primaryStage.setScene(new Scene(root));
        primaryStage.show();
    }


    
    
    
    
}
