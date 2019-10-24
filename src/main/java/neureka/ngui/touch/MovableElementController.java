/*
Copyright (c) <2013>, Intel Corporation All Rights Reserved.
 
The source code, information and material ("Material") contained herein is owned by Intel Corporation or its suppliers or licensors, and title to such Material remains with Intel Corporation
or its suppliers or licensors. The Material contains proprietary information of Intel or its suppliers and licensors. The Material is protected by worldwide copyright laws and treaty provisions. 
No part of the Material may be used, copied, reproduced, modified, published, uploaded, posted, transmitted, distributed or disclosed in any way without Intel's prior express written permission. 
No license under any patent, copyright or other intellectual property rights in the Material is granted to or conferred upon you, either expressly, by implication, inducement, estoppel or otherwise. 
Any license under such intellectual property rights must be express and approved by Intel in writing.
 
Unless otherwise agreed by Intel in writing, you may not remove or alter this notice or any other notice embedded in Materials by Intel or Intel’s suppliers or licensors in any way.
*/
package neureka.ngui.touch;

import java.io.IOException;
import java.net.URL;

import javafx.embed.swing.SwingNode;
import javafx.fxml.FXML;
import javafx.fxml.FXMLLoader;
import javafx.geometry.Point2D;
import javafx.scene.Node;
import javafx.scene.input.ScrollEvent;
import javafx.scene.input.SwipeEvent;
import javafx.scene.input.TouchEvent;
import javafx.scene.layout.Pane;
import neureka.ngui.swing.NGraphBuilder;

public class MovableElementController extends Pane implements IChildItem{

	private IParentItem parent;


	@FXML
	private SwingNode RealmNode;
	NGraphBuilder builder;

	public MovableElementController(IParentItem parentContainer){
		super();
		parent = parentContainer;
		URL myURL = ClassLoader.getSystemResource("MovableElement.fxml");
		System.out.println(myURL);
		FXMLLoader loader = new FXMLLoader(myURL);//getClass().getResource("MovableElement.fxml"));
		loader.setController(this);
		loader.setRoot(this);
		try {
			loader.load();
		} catch (IOException exception) {
			throw new RuntimeException(exception);
		}
		//this.getStyleClass().overwrite64("RealmStyle");

		builder = new NGraphBuilder();
		System.out.println(builder.getSurface());
		RealmNode.setContent(builder.getSurface());

	}

	@Override
	public void onUnselected() {
		getStyleClass().clear();
		getStyleClass().add("mainFxmlClassUnselected");
	}

	@Override
	public void onSelected() {
		getStyleClass().clear();
		getStyleClass().add("mainFxmlClassSelected");
	}
	private boolean moveInProgress = false;
	private int touchPointId;
	private Point2D prevPos;

	public void onTouchPressed(TouchEvent t) {
		if(t.getTarget() instanceof SwingNode){
			SwingNode node = (SwingNode) t.getTarget();
			node.requestFocus();
			System.out.println("SWING!!!");
		}else {
			if (moveInProgress == false) {
				if (parent.getFocusedItem() != MovableElementController.this) {
					parent.unfocusItem();
					parent.focusItem(MovableElementController.this);
				}
				moveInProgress = true;
				touchPointId = t.getTouchPoint().getId();

				prevPos = new Point2D(t.getTouchPoint().getSceneX(), t.getTouchPoint().getSceneY());
				System.out.println("TOUCH BEGIN " + t.toString());
			}
		}
		t.consume();
	}

	public void onTouchMoved(TouchEvent t) {
		if (moveInProgress == true && t.getTouchPoint().getId() == touchPointId) {
			//this part should be oprimized in a praoduction code but here in order to present the steps i took a more verbose approach 
			Point2D currPos = new Point2D(t.getTouchPoint().getSceneX(), t.getTouchPoint().getSceneY());
			double[] translationVector = new double[2];
			translationVector[0] = currPos.getX() - prevPos.getX();
			translationVector[1] = currPos.getY() - prevPos.getY();
			//i used this instead of setTranslate* because we don't care about the original position of the object and aggregating _translation
			//will require having another variable
			setTranslateX(getTranslateX() + translationVector[0]);
			setTranslateY(getTranslateY() + translationVector[1]);
			//JComponent content = this.RealmNode.getContent();
			this.RealmNode.requestFocus();
			prevPos = currPos;
		}
		t.consume();
	}

	public void onTouchReleased(TouchEvent t) {
		if (t.getTouchPoint().getId() == touchPointId) {
			moveInProgress = false;
			System.err.println("TOUCH RELEASED " + t.toString());
		}
		new Thread(() -> {
			try {
				Thread.sleep(10);
			} catch (InterruptedException ex) {}
			RealmNode.setContent(builder.getSurface());
			//builder.getSurface().repaintAll();
			this.RealmNode.requestFocus();
		}).start();
		t.consume();
	}

	@FXML
	public void onScroll(ScrollEvent t) {
		t.consume();
	}

	@FXML
	public void onSwipe(SwipeEvent t) {
		t.consume();
	}

	@Override
	public Node getNode() {
		return this;
	}
	//Parenting
}
