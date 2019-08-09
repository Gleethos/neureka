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

import java.net.URL;
import java.util.ResourceBundle;
import javafx.animation.TranslateTransition;
import javafx.beans.value.ChangeListener;
import javafx.beans.value.ObservableValue;
import javafx.event.ActionEvent;
import javafx.fxml.FXML;
import javafx.fxml.Initializable;
import javafx.scene.Node;
import javafx.scene.control.Toggle;
import javafx.scene.control.ToggleButton;
import javafx.scene.control.ToggleGroup;
import javafx.scene.input.RotateEvent;
import javafx.scene.input.ScrollEvent;
import javafx.scene.input.SwipeEvent;
import javafx.scene.input.ZoomEvent;
import javafx.scene.layout.HBox;
import javafx.scene.layout.Pane;
import javafx.util.Duration;

public class TouchPaneController implements Initializable, IParentItem {

	public enum GestureSelection {
		SWIPE, SCROLL
	}
	@FXML private Pane touchPane;
	@FXML private HBox buttons;
	@FXML ToggleButton setScrollBtn;
	@FXML ToggleButton setSwipeBtn;
	@FXML ToggleGroup gestureSelectionGroup;

	private IChildItem currentSelection;
	private GestureSelection selectedGesture = GestureSelection.SCROLL;

	@FXML
	private void handleResetScene(ActionEvent event) {
		touchPane.getChildren().clear();
		touchPane.getChildren().add(buttons);
	}

	@FXML
	private void handleAddElement(ActionEvent event) {

		MovableElementController elem = new MovableElementController(this);
		elem.setTranslateX(touchPane.getWidth() / 2.0);
		elem.setTranslateY(touchPane.getHeight() / 2.0);
		elem.setScaleX(0.1);
		elem.setScaleY(0.1);
		touchPane.getChildren().remove(buttons);
		touchPane.getChildren().add(elem);
		touchPane.getChildren().add(buttons);
	}

	@Override
	public void initialize(URL url, ResourceBundle rb) {
		setScrollBtn.setUserData(GestureSelection.SCROLL);
		setSwipeBtn.setUserData(GestureSelection.SWIPE);

		gestureSelectionGroup.selectedToggleProperty().addListener(new ChangeListener<Toggle>() {
			@Override
			public void changed(ObservableValue<? extends Toggle> ov, Toggle oldValue, Toggle newValue) {
				selectedGesture = (GestureSelection) newValue.getUserData();
			}
		});
	}

	@Override
	public void focusItem(IChildItem rect) {
		if (rect == null) {
			throw new NullPointerException();
		}

		if (!touchPane.getChildren().contains(rect)) {
			throw new Error("Trying to register an element that is not a child of this pane");
		} else {

			Node rectNode = rect.getNode();
			currentSelection = rect;
			currentSelection.onSelected();
			//pushing to the end of the list to make it appear on the top
			touchPane.getChildren().remove(rectNode);
			touchPane.getChildren().remove(buttons);
			touchPane.getChildren().add(rectNode);
			touchPane.getChildren().add(buttons);
		}
	}

	@Override
	public boolean isFocusing() {
		return (currentSelection != null);
	}

	@Override
	public IChildItem getFocusedItem() {
		return currentSelection;
	}

	@Override
	public void unfocusItem() {
		if (currentSelection != null) {
			currentSelection.onUnselected();
		}
		currentSelection = null;
	}

	@FXML
	public void onScroll(ScrollEvent t) {
		if (selectedGesture == GestureSelection.SCROLL && currentSelection != null) {
			Node selNode = currentSelection.getNode();
			selNode.setTranslateX(selNode.getTranslateX() + (t.getDeltaX() / 10.0));
			selNode.setTranslateY(selNode.getTranslateY() + (t.getDeltaY() / 10.0));
		}
		t.consume();
	}

	@FXML
	public void onZoom(ZoomEvent t) {
		if (currentSelection != null) {
			Node selNode = currentSelection.getNode();
			selNode.setScaleX(selNode.getScaleX() * t.getZoomFactor());
			selNode.setScaleY(selNode.getScaleY() * t.getZoomFactor());
			if(selNode instanceof  Pane){
				((Pane) selNode).getChildren().forEach((n)->{});
			}
		}

		t.consume();
	}

	@FXML
	public void onRotate(RotateEvent t) {
		if (currentSelection != null) {
			Node selNode = currentSelection.getNode();
			selNode.setRotate(selNode.getRotate() + t.getAngle());
		}

		t.consume();
	}

	@FXML
	public void onSwipe(SwipeEvent t) {
		if (selectedGesture == GestureSelection.SWIPE && currentSelection != null) {
			Node selNode = currentSelection.getNode();
			TranslateTransition transition = new TranslateTransition(Duration.millis(1000), selNode);
			if (t.getEventType() == SwipeEvent.SWIPE_DOWN) {
				transition.setByY(100);
			} else if (t.getEventType() == SwipeEvent.SWIPE_UP) {
				transition.setByY(-100);
			} else if (t.getEventType() == SwipeEvent.SWIPE_LEFT) {
				transition.setByX(-100);
			} else if (t.getEventType() == SwipeEvent.SWIPE_RIGHT) {
				transition.setByX(100);
			}
			transition.play();

		}
		t.consume();
	}
}
