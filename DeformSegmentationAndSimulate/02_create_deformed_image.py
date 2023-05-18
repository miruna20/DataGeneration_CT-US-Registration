import argparse
import os

import numpy as np
import pyvista as pv
import torchio
from scipy.interpolate import Rbf, griddata
from scipy.interpolate import RBFInterpolator
import torchio as tio
import SimpleITK as sitk


def get_range_from_image(spine_image):
    img = sitk.ReadImage(spine_image)
    origin = np.array(img.GetOrigin())
    final = np.array(img.GetDirection()).reshape((3,3)) @ np.array(img.GetSize()) * np.array(img.GetSpacing()) + origin
    c = np.stack((origin, final))
    origin = c.min(axis=0)
    final = c.max(axis=0)
    return origin, final

def calculate_deformation(non_deformed_mesh, deformed_mesh_name, spine_image, sampling_freq, subfolder_path):

    origin, final = get_range_from_image(spine_image)
    ddf_spacing = [5, 5, 5]
    non_deformed_mesh = pv.read(non_deformed_mesh)
    deformed_mesh = pv.read(os.path.join(subfolder_path, deformed_mesh_name))

    x_grid = np.mgrid[
                        origin[0]:final[0]:ddf_spacing[0],
                        origin[1]:final[1]:ddf_spacing[1],
                        origin[2]:final[2]:ddf_spacing[2]]

    x_flat = x_grid.reshape(3, -1).T

    # node_def = np.zeros_like(non_deformed_mesh.points.shape)
    node_def = non_deformed_mesh.points - deformed_mesh.points

    def_grid_x = interp_1d_deformation(node_def, non_deformed_mesh, x_flat, x_grid, axis=0, sampling_freq=sampling_freq)
    def_grid_y = interp_1d_deformation(node_def, non_deformed_mesh, x_flat, x_grid, axis=1, sampling_freq=sampling_freq)
    def_grid_z = interp_1d_deformation(node_def, non_deformed_mesh, x_flat, x_grid, axis=2, sampling_freq=sampling_freq)

    full_grid = np.stack([def_grid_x, def_grid_y, def_grid_z], axis=3).astype(np.float64)

    full_grid = np.transpose(full_grid, axes=[2, 1, 0, 3])

    img = sitk.GetImageFromArray(full_grid)
    img.SetOrigin(origin)
    img.SetSpacing(ddf_spacing)

    field_path = os.path.join(subfolder_path, deformed_mesh_name.replace(".obj", "_field.mha"))
    sitk.WriteImage(img, field_path)
    print("deformation saved")

    return field_path


def interp_1d_deformation(node_def, non_deformed_mesh, x_flat, x_grid, axis, sampling_freq=50):
    interp = RBFInterpolator(non_deformed_mesh.points[::sampling_freq, :], node_def[::sampling_freq, axis])
    y_flat = interp(x_flat)
    y_grid = np.reshape(y_flat, list(x_grid.shape)[1:])
    return y_grid


def process(txt_file, root_path_spine, segmentation_folder, sampling_freq):
    # Read the lines from the text file
    with open(txt_file, 'r') as file:
        lines = file.readlines()

    # Process each line
    for line in lines:
        line = line.strip()  # Remove leading/trailing whitespaces
        subfolder_path = os.path.join(root_path_spine, f"sub-{line}/")

        # Find the non-deformed mesh
        non_deformed_files = [
            file_name
            for file_name in os.listdir(subfolder_path)
            if file_name.endswith(
                '.obj') and 'centered' not in file_name and 'scaled' not in file_name and 'deformed' not in file_name
        ]

        deformed_files = [
            file_name
            for file_name in os.listdir(subfolder_path)
            if file_name.endswith(
                '.obj') and 'centered' not in file_name and 'scaled' not in file_name and 'deformed' in file_name
        ]

        seg_files = [
            file_name
            for file_name in os.listdir(subfolder_path)
            if file_name.endswith('.nii.gz') and 'crop' in file_name and 'seg' in file_name and line in file_name
        ]

        ct_files = [
            file_name
            for file_name in os.listdir(subfolder_path)
            if file_name.endswith('.nii.gz') and 'crop' in file_name and 'seg' not in file_name and line in file_name
        ]

        if len(deformed_files) and len(non_deformed_files) and len(seg_files) and len(ct_files):
            non_deformed_file = os.path.join(subfolder_path, non_deformed_files[0])
            seg_file = os.path.join(subfolder_path, seg_files[0])
            ct_file = os.path.join(subfolder_path, ct_files[0])
            for deformed_file in deformed_files:
                field_path = calculate_deformation(non_deformed_mesh=non_deformed_file, deformed_mesh_name=deformed_file,
                                            spine_image=ct_file, sampling_freq=sampling_freq, subfolder_path=subfolder_path)
                apply_deformation(ct_file, seg_file, field_path, "./imfusion_workspaces/apply_deformation.iws")


def apply_deformation(ct_file, seg_file, field_path, workspace_path):

    # apply deformation on CT image
    arguments_imfusion = {"INPUTIMAGE": ct_file, "INPUTFIELD": field_path, "NEARESTINTERP": 0,
                          "OUTPUTIMAGE": field_path.replace("_field.mha", "_ct_img.nii.gz")}
    arguments_imfusion = ' '.join([f"{key}={value}" for key, value in arguments_imfusion.items()])

    print('ARGUMENTS: ', arguments_imfusion)
    os.system("ImFusionConsole" + " " + workspace_path + " " + arguments_imfusion)
    print('################################################### ')

    # apply deformation on segmentation label
    arguments_imfusion = {"INPUTIMAGE": seg_file, "INPUTFIELD": field_path, "NEARESTINTERP": 1,
                          "OUTPUTIMAGE": field_path.replace("_field.mha", "_seg.nii.gz")}
    arguments_imfusion = ' '.join([f"{key}={value}" for key, value in arguments_imfusion.items()])

    print('ARGUMENTS: ', arguments_imfusion)
    os.system("ImFusionConsole" + " " + workspace_path + " " + arguments_imfusion)
    print('################################################### ')


if __name__ == "__main__":
    """
        # example setup
        root_vertebrae = "/home/miruna20/Documents/Thesis/sofa/vertebrae/train"
        spine_name = "sub-verse500"
        txt_file = "../samples/test.txt"
    """

    arg_parser = argparse.ArgumentParser(description="Extrapolate the deformation field to cover the whole image")

    arg_parser.add_argument(
        "--sampling_freq",
        default=50,
        help="sampling rate from the meshes, default=50"
    )

    arg_parser.add_argument(
        "--root_path_spine",
        required=True,
        dest="root_path_spine",
        help="Root path of the spine folders"
    )

    arg_parser.add_argument(
        "--list_file_names",
        required=True,
        dest="txt_file",
        help="Txt file that contains all spines that contain all lumbar vertebrae"
    )

    arg_parser.add_argument(
        "--segmentation_folder",
        required=True,
        help="folder containing the output of the total segmentator"
    )

    print("Calculating the deformation over the whole image")
    #
    args = arg_parser.parse_args()

    process(txt_file=args.txt_file, root_path_spine=args.root_path_spine,
            segmentation_folder=args.segmentation_folder, sampling_freq=args.sampling_freq)
    # calculate_deformation(non_deformed_mesh=args.non_deformed_mesh, deformed_mesh=args.deformed_mesh,
    #                       spine_image=args.spine_image, sampling_freq=args.sampling_freq)
