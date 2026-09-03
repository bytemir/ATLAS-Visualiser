import uproot
import numpy as np
import pyvista as pv

def LoadEvent(file_path, event_index=0):
    with uproot.open(file_path) as f:
        tree = f["mini"]
        
        num_entries = tree.num_entries
        if event_index < 0 or event_index >= num_entries:
            raise IndexError(f"Event index {event_index} out of range (0 to {num_entries - 1}).")
            
        branches = ["lep_pt", "lep_eta", "lep_phi", "lep_charge", "lep_type", "met_et", "met_phi"]
        data = tree.arrays(branches, entry_start=event_index, entry_stop=event_index+1, library="np")
        
    EventData = {
        "pt": data["lep_pt"][0] / 1000.0,     # Transverse Momentum (GeV)
        "eta": data["lep_eta"][0],            # Pseudorapidity 
        "phi": data["lep_phi"][0],            # Azimuthal Angle 
        "charge": data["lep_charge"][0],      # Charge
        "pdgId": data["lep_type"][0],         # PDG Identifier
        "met_et": data["met_et"][0] / 1000.0, # Missing Transverse Energy (GeV)
        "met_phi": data["met_phi"][0]         # Missing Transverse Energy Angle
    }
    return EventData

def CoordinateTransformation(EventData):
    Pseudorapidity = EventData["eta"]
    TransverseMomentum = EventData["pt"]
    AzimuthalAngle = EventData["phi"]
    
    PolarAngle = 2.0 * np.arctan(np.exp(-Pseudorapidity))
    
    MomentumX = TransverseMomentum * np.cos(AzimuthalAngle)
    MomentumY = TransverseMomentum * np.sin(AzimuthalAngle)
    MomentumZ = TransverseMomentum * np.sinh(Pseudorapidity)
    TotalMomentum = np.sqrt(MomentumX**2 + MomentumY**2 + MomentumZ**2)
    
    UnitX = np.sin(PolarAngle) * np.cos(AzimuthalAngle)
    UnitY = np.sin(PolarAngle) * np.sin(AzimuthalAngle)
    UnitZ = np.cos(PolarAngle)
    
    MissingEnergyX = EventData["met_et"] * np.cos(EventData["met_phi"])
    MissingEnergyY = EventData["met_et"] * np.sin(EventData["met_phi"])
    
    TransformedData = {
        "PolarAngle": PolarAngle,
        "MomentumComponents": np.column_stack((MomentumX, MomentumY, MomentumZ)),
        "TotalMomentum": TotalMomentum,
        "UnitDirections": np.column_stack((UnitX, UnitY, UnitZ)),
        "MissingEnergyVector": np.array([MissingEnergyX, MissingEnergyY, 0.0])
    }
    return TransformedData

def CalculateStraightBeamPaths(ProgressFraction, BeamLength=12.0, NumPoints=20):
    CurrentZ = BeamLength * (1.0 - ProgressFraction)
    
    z1 = np.linspace(BeamLength, CurrentZ, NumPoints)
    z2 = np.linspace(-BeamLength, -CurrentZ, NumPoints)
    
    Proton1Path = np.zeros((NumPoints, 3))
    Proton1Path[:, 2] = z1
    
    Proton2Path = np.zeros((NumPoints, 3))
    Proton2Path[:, 2] = z2
    
    return Proton1Path, Proton2Path

def CreateDetectorGeometry(InnerRadius=2.0, OuterRadius=4.0, Length=8.0):
    InnerCylinder = pv.Cylinder(center=(0, 0, 0), direction=(0, 0, 1), radius=InnerRadius, height=Length, resolution=40)
    OuterCylinder = pv.Cylinder(center=(0, 0, 0), direction=(0, 0, 1), radius=OuterRadius, height=Length, resolution=40)
    return InnerCylinder, OuterCylinder

def CalculateAnimatedTrajectories(EventData, TransformedData, ProgressFraction, MagneticField=2.0, DetectorRadius=4.0, DetectorLength=8.0, NumPoints=40):
    TransverseMomentum = EventData["pt"]
    Charge = EventData["charge"]
    AzimuthalAngle = EventData["phi"]
    Pseudorapidity = EventData["eta"]
    
    NumParticles = len(TransverseMomentum)
    OutgoingPaths = []
    ZMax = DetectorLength / 2.0

    for i in range(NumParticles):
        pt = TransverseMomentum[i]
        q = Charge[i]
        phi = AzimuthalAngle[i]
        eta = Pseudorapidity[i]
        
        R = pt / (0.3 * abs(q) * MagneticField)
        xc = -q * R * np.sin(phi)
        yc =  q * R * np.cos(phi)
        
        CosTerm = np.clip(1.0 - (DetectorRadius**2) / (2.0 * R**2), -1.0, 1.0)
        t_radial_max = np.arccos(CosTerm)
        
        sinh_eta = np.sinh(eta)
        t_z_max = ZMax / (R * abs(sinh_eta)) if abs(sinh_eta) > 1e-5 else t_radial_max
        
        t_max_final = min(t_radial_max, t_z_max)
        t_current = t_max_final * ProgressFraction
        t = np.linspace(0, t_current, NumPoints)
        
        x = xc + q * R * np.sin(phi + t)
        y = yc - q * R * np.cos(phi + t)
        z = R * t * sinh_eta
        
        OutgoingPaths.append(np.column_stack((x, y, z)))
        
    METVector = TransformedData["MissingEnergyVector"]
    METMagnitude = np.linalg.norm(METVector[:2])
    
    if METMagnitude > 0:
        UnitMET = METVector / METMagnitude
        METPath = np.outer(np.linspace(0, ProgressFraction, NumPoints), UnitMET * DetectorRadius)
    else:
        METPath = np.zeros((NumPoints, 3))
    
    return {
        "OutgoingPaths": np.array(OutgoingPaths),
        "METPath": METPath
    }

def GetCurrentTrackLabels(EventData, TransformedData, Trajectories):
    OutgoingPaths = Trajectories["OutgoingPaths"]
    PDGIDs = EventData["pdgId"]
    Charges = EventData["charge"]
    PTs = EventData["pt"]
    Etas = EventData["eta"]
    Phis = EventData["phi"]
    TotalMomenta = TransformedData["TotalMomentum"]
    
    CurrentPoints = []
    Labels = []
    
    for Index, Path in enumerate(OutgoingPaths):
        CurrentTip = Path[-1]
        pdg = PDGIDs[Index]
        charge = Charges[Index]
        pt = PTs[Index]
        eta = Etas[Index]
        phi = Phis[Index]
        p_total = TotalMomenta[Index]
        
        ChargeSign = "+" if charge > 0 else "-"
        ParticleType = "e" if abs(pdg) == 11 else "mu" if abs(pdg) == 13 else "Track"
        
        CurrentPoints.append(CurrentTip)
        Labels.append(
            f"{ParticleType}{ChargeSign} [q={charge:+d}]\n"
            f"p={p_total:.1f} GeV | pT={pt:.1f} GeV\n"
            f"eta={eta:.2f} | phi={phi:.2f}\n"
            f"Pos: ({CurrentTip[0]:.1f}, {CurrentTip[1]:.1f}, {CurrentTip[2]:.1f})"
        )
        
    METPath = Trajectories["METPath"]
    METTip = METPath[-1]
    
    CurrentPoints.append(METTip)
    Labels.append(
        f"MET (Missing ET)\n"
        f"ET={EventData['met_et']:.1f} GeV | phi={EventData['met_phi']:.2f}\n"
        f"Pos: ({METTip[0]:.1f}, {METTip[1]:.1f}, {METTip[2]:.1f})"
    )
    
    return np.array(CurrentPoints), Labels