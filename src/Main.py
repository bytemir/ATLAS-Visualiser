import pyvista as pv
import numpy as np
import tkinter as tk
from tkinter import simpledialog
import DataExtraction

def ConstructPolyLine(points):
    n_points = len(points)
    cells = np.hstack([[n_points], np.arange(n_points)])
    return pv.PolyData(points, lines=cells)

def StartDisplaySession(file_path):
    Plotter = pv.Plotter(window_size=[1280, 720])
    Plotter.set_background("#0a0a0f")
    Plotter.enable_anti_aliasing("fxaa")
    
    Plotter.add_axes(
        line_width=1,
        color="white",
        x_color="#ffffff",
        y_color="#ffffff",
        z_color="#ffffff"
    )
    
    AxisLength = 16.0
    Plotter.add_lines(np.array([[-AxisLength, 0, 0], [AxisLength, 0, 0]]), color="#ffffff", width=1)
    Plotter.add_lines(np.array([[0, -AxisLength, 0], [0, AxisLength, 0]]), color="#ffffff", width=1)
    Plotter.add_lines(np.array([[0, 0, -AxisLength], [0, 0, AxisLength]]), color="#ffffff", width=1)

    InnerCyl, OuterCyl = DataExtraction.CreateDetectorGeometry(InnerRadius=2.0, OuterRadius=4.0, Length=8.0)
    Plotter.add_mesh(InnerCyl, color="#00aaff", opacity=0.12, show_edges=True, edge_color="#0055aa", line_width=1)
    Plotter.add_mesh(OuterCyl, color="#00ffcc", opacity=0.08, show_edges=True, edge_color="#00aa88", line_width=1)

    VertexMesh = pv.Sphere(radius=0.15, center=(0, 0, 0))
    VertexActor = Plotter.add_mesh(VertexMesh, color="#ffffff", emissive=True, opacity=0.0)

    # Params
    State = {
        "event_index": 0,
        "frames_per_phase": 50,
        "particle_actors": [],
        "proton_labels": None,
        "track_labels": None,
        "ui_text_actor": None,
        "is_animating": False
    }

    def RefreshOverlay():
        if State["ui_text_actor"] is not None:
            Plotter.remove_actor(State["ui_text_actor"])
            
        hud_text = (
            f"Event Index: {State['event_index']}  (Press 'E' to change)\n"
            f"Frames/Phase: {State['frames_per_phase']} (Press 'F' to change)\n"
            f"Press 'R' to Replay Animation"
        )
        State["ui_text_actor"] = Plotter.add_text(
            hud_text, position="upper_left", font_size=10, color="white", shadow=False
        )

    def RenderSequence():
        if State["is_animating"]:
            return
            
        State["is_animating"] = True
        
        try:
            EventData = DataExtraction.LoadEvent(file_path, event_index=State["event_index"])
        except Exception as e:
            print(f"Error loading event: {e}")
            State["is_animating"] = False
            return

        TransformedData = DataExtraction.CoordinateTransformation(EventData)
        FramesPerPhase = State["frames_per_phase"]
        BeamLength = 12.0

        # Cleanup
        for actor in State["particle_actors"]:
            Plotter.remove_actor(actor)
        State["particle_actors"].clear()

        if State["proton_labels"] is not None:
            Plotter.remove_actor(State["proton_labels"])
            State["proton_labels"] = None
            
        if State["track_labels"] is not None:
            Plotter.remove_actor(State["track_labels"])
            State["track_labels"] = None

        DummyPoints = np.zeros((20, 3))
        
        Beam1Poly = ConstructPolyLine(DummyPoints)
        Beam2Poly = ConstructPolyLine(DummyPoints)
        Beam1Actor = Plotter.add_mesh(Beam1Poly, color="#00ffff", line_width=4, render_lines_as_tubes=True)
        Beam2Actor = Plotter.add_mesh(Beam2Poly, color="#00ffff", line_width=4, render_lines_as_tubes=True)
        State["particle_actors"].extend([Beam1Actor, Beam2Actor])

        NumParticles = len(EventData["pt"])
        ParticlePolys = []
        for i in range(NumParticles):
            color = "#ff0055" if abs(EventData["pdgId"][i]) == 11 else "#00ff66"
            poly = ConstructPolyLine(np.zeros((40, 3)))
            actor = Plotter.add_mesh(poly, color=color, line_width=4, render_lines_as_tubes=True)
            ParticlePolys.append(poly)
            State["particle_actors"].append(actor)

        METPoly = ConstructPolyLine(np.zeros((40, 3)))
        METActor = Plotter.add_mesh(METPoly, color="#ff00ff", line_width=3, render_lines_as_tubes=True)
        State["particle_actors"].append(METActor)

        VertexActor.GetProperty().SetOpacity(0.0)
        RefreshOverlay()

        # inc protons
        for Step in range(FramesPerPhase + 1):
            Progress = Step / FramesPerPhase
            Proton1_Path, Proton2_Path = DataExtraction.CalculateStraightBeamPaths(Progress, BeamLength, NumPoints=20)
            
            Beam1Poly.points = Proton1_Path
            Beam2Poly.points = Proton2_Path
            
            Tip1, Tip2 = Proton1_Path[0], Proton2_Path[0]
            
            if State["proton_labels"] is not None:
                Plotter.remove_actor(State["proton_labels"])
                
            Proton1_Label = f"Proton (p+)\nq=+1 | Pos: (0.0, 0.0, {Tip1[2]:.1f})"
            Proton2_Label = f"Proton (p+)\nq=+1 | Pos: (0.0, 0.0, {Tip2[2]:.1f})"
            
            State["proton_labels"] = Plotter.add_point_labels(
                np.array([Tip1, Tip2]),
                [Proton1_Label, Proton2_Label],
                show_points=False,
                font_size=10,
                text_color="#00ffff",
                shadow=False,
                shape=None,
                always_visible=True
            )
            
            Plotter.render()

        Beam1Actor.SetVisibility(False)
        Beam2Actor.SetVisibility(False)
        if State["proton_labels"] is not None:
            Plotter.remove_actor(State["proton_labels"])
            State["proton_labels"] = None

        # collision event
        VertexActor.GetProperty().SetOpacity(1.0)
        for _ in range(4):
            Plotter.render()

        # outgoing particles
        for Step in range(1, FramesPerPhase + 1):
            Progress = Step / FramesPerPhase
            
            Trajectories = DataExtraction.CalculateAnimatedTrajectories(
                EventData, TransformedData, ProgressFraction=Progress, DetectorRadius=4.0, DetectorLength=8.0, NumPoints=40
            )
            
            for i, Path in enumerate(Trajectories["OutgoingPaths"]):
                ParticlePolys[i].points = Path
                
            METPoly.points = Trajectories["METPath"]
            
            if State["track_labels"] is not None:
                Plotter.remove_actor(State["track_labels"])
                
            CurrentPoints, Labels = DataExtraction.GetCurrentTrackLabels(EventData, TransformedData, Trajectories)
            State["track_labels"] = Plotter.add_point_labels(
                CurrentPoints,
                Labels,
                show_points=False,
                point_size=0,
                font_size=10,
                text_color="white",
                shadow=False,
                shape=None,
                always_visible=True
            )
            
            Plotter.render()

        State["is_animating"] = False

    def RequestEventIndex():
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        val = simpledialog.askinteger("Input", "Enter Event Index:", initialvalue=State["event_index"], minvalue=0)
        root.destroy()
        if val is not None:
            State["event_index"] = val
            RenderSequence()

    def RequestFrameSpeed():
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        val = simpledialog.askinteger("Input", "Enter Frames Per Phase:", initialvalue=State["frames_per_phase"], minvalue=1)
        root.destroy()
        if val is not None:
            State["frames_per_phase"] = val
            RenderSequence()

    Plotter.add_key_event("e", RequestEventIndex)
    Plotter.add_key_event("f", RequestFrameSpeed)
    Plotter.add_key_event("r", RenderSequence)

    Plotter.camera_position = [(16, 14, 10), (0, 0, 0), (0, 0, 1)]
    
    Plotter.show(interactive_update=True, title="ATLAS Detector | Event Simulation")
    
    RenderSequence()

    while not Plotter.render_window.GetNeverRendered():
        Plotter.update()

def Execute():
    FilePath = "data/data_A.4lep.root"
    StartDisplaySession(FilePath)

if __name__ == "__main__":
    Execute()