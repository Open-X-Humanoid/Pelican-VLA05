"""Embodiment registry — one entry per dataset/embodiment (single source of
truth for "what does the arm/dataset look like").

Consumers:
  - `sources/libero.py` / `sources/robotwin.py` — camera keys / state layout
  - `sources/robomind.py` + `sources/_lerobot_frame.py` — lerobot dataset config
  - `adapters/pelicanvla.py::_resolve_state_stats` — AgileX state field order

Add a new embodiment = append a new `Embodiment` to `EMBODIMENTS` (and, if the
loader is `"lerobot"`, fill `tasks` / `state_cols` / camera keys).

Pure Python, importable from any layer. Dataset roots are pulled from
`attnvis.config.DATA_ROOTS` (override via `ATTNVIS_*_ROOT`).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from attnvis.config import DATA_ROOTS

_U  = str(DATA_ROOTS["unify"])
_V3 = str(DATA_ROOTS["lerobot_v3"])
_VL = str(DATA_ROOTS["vla_lerobot"])


@dataclass
class Camera:
    """One camera of an embodiment. `short` = the name used in `Frame.cams`.

    role: "third_person" | "wrist" | "aux".
    """
    short: str
    key: str = ""
    role: str = "third_person"


@dataclass
class Embodiment:
    """Complete description of one dataset's embodiment (single source of truth)."""
    name: str                          # matches the source/dataset short name
    arm: str                           # "single" | "dual"
    state_dim: int
    cameras: list[Camera]
    loader: str                        # "hdf5_libero" | "hdf5_robotwin" | "lerobot"
    state_cols: list[tuple[str, int]] | None = None    # lerobot: parquet columns → state
    instruction_source: str | None = None              # "episode_tasks" | None
    tasks: dict[str, str] = field(default_factory=dict)  # suite → lerobot dataset root
    default_suite: str | None = None
    note: str = ""

    @property
    def cams(self) -> list[str]:
        return [c.short for c in self.cameras]

    @property
    def camera_keys(self) -> dict[str, str]:
        return {c.short: c.key for c in self.cameras}

    def third_person(self) -> list[str]:
        return [c.short for c in self.cameras if c.role == "third_person"]

    def wrist(self) -> list[str]:
        return [c.short for c in self.cameras if c.role == "wrist"]


# ── LeRobot dataset roots per suite ──────────────────────────────────────────
_UR5E_G = f"{_U}/ur_5e_singleArm-gripper-3cameras_2_new-gripper"
UR5E_TASKS = {
    "install_gears": f"{_UR5E_G}/ur5e_singleArm-gripper-3cameras_2_Install-gears_20260416_pm/lerobot_RoboMIND",
    "place_usb":     f"{_UR5E_G}/ur5e_singleArm-gripper-3cameras_2place_usb_male_prev_20260423_pm/lerobot_RoboMIND",
    "place_wp":      f"{_UR5E_G}/ur5e_singleArm-gripper-3cameras_2_place_wp_male_prev_20260423_am/lerobot_RoboMIND",
    "place_bnc":     f"{_UR5E_G}/ur5e_singleArm-gripper-3cameras_2_place_bnc_male_prev_260420_am/lerobot_RoboMIND",
    "place_dsub":    f"{_UR5E_G}/ur5e_singleArm-gripper-3cameras_2_place_dsub_male_prev_260418_am/lerobot_RoboMIND",
    "place_rj45":    f"{_UR5E_G}/ur5e_singleArm-gripper-3cameras_2_place_rj45_male_prev_20260506_pm/lerobot_RoboMIND",
    "unplug_usb":    f"{_UR5E_G}/ur5e_singleArm-gripper-3cameras_2_unplug_front_usb_20260416_am/lerobot_RoboMIND",
}

_TK_S6  = f"{_U}/tienkung_soft_gripper/tienkung_station_dualArm-gripper-3cameras_6"
_TK_S23 = f"{_U}/tienkung_soft_gripper/tienkung_station_dualArm-gripper-3cameras_23"
TIENKUNG_TASKS = {
    "organize_tools": f"{_TK_S23}/tienkung_station_dualArm-gripper-3cameras_23_organize-cleaning-tools-on-the-pegboard_20260423_pm/success/lerobot_RoboMIND",
    "sweep":          f"{_TK_S23}/tienkung_station_dualArm-gripper-3cameras_23_sweep_table_trash_dustpan_20260420_pm/success/lerobot_RoboMIND",
    "pack_carton":    f"{_TK_S6}/tienkung_station_dualArm-gripper-3cameras_6_pack_in_cartron_seal_20260423_pm/success/lerobot_RoboMIND",
    "filter_liquid":  f"{_TK_S6}/tienkung_station_dualArm-gripper-3cameras_6_filter-liquid-using-a-funnel_20260423_am/success/lerobot_RoboMIND",
}

_V3_S1 = f"{_V3}/ur_5e/ur_5e_singleArm-gripper-4cameras_1"
_V3_S2 = f"{_V3}/ur_5e/ur_5e_singleArm-gripper-4cameras_2"
UR5E_V3_TASKS = {
    "pick_bread":     f"{_V3_S2}/pick_bread_250623/success/lerobot_RoboMIND",
    "push_button":    f"{_V3_S2}/push_button_250625/success/lerobot_RoboMIND",
    "open_toaster":   f"{_V3_S2}/open_toaster_250623/success/lerobot_RoboMIND",
    "hang_scissors":  f"{_V3_S2}/hang_scissors/success/lerobot_RoboMIND",
    "place_banana":   f"{_V3_S2}/pick_and_place_banana/success/lerobot_RoboMIND",
    "put_bread":      f"{_V3_S2}/put_bread_250623/success/lerobot_RoboMIND",
    "place_spoon":    f"{_V3_S2}/pick_and_place_spoon/success/lerobot_RoboMIND",
    "open_door":      f"{_V3_S2}/open_the_door/success/lerobot_RoboMIND",
    "bread_to_trash": f"{_V3_S2}/pick_bread_man_into_trash/success/lerobot_RoboMIND",
    "pour_water":     f"{_V3_S1}/gpt_test_ur_single_4cam_1_pour_water_into_blue_cup/success/lerobot_RoboMIND",
    "stir":           f"{_V3_S1}/gpt_ur5e_singleArm_stir/success/lerobot_RoboMIND",
}

_V3_DUAL = f"{_V3}/ur_5e/ur_5e_dualArm-gripper-6cameras_1"
UR5E_DUAL_TASKS = {
    "add_ice": f"{_V3_DUAL}/ur_01_add_ice_20250708/success/lerobot_RoboMIND",
}

_REALSRC = f"{_VL}/realsource_world_v30"
REALSOURCE_V30_TASKS = {
    "arrange_the_cups":                  f"{_REALSRC}/RealSource__Arrange_the_cups",
    "cable_plugging_able":               f"{_REALSRC}/RealSource__Cable_Plugging_able",
    "clean_the_convenience_store":       f"{_REALSRC}/RealSource__Clean_the_convenience_store",
    "cook_rice_using_an_electric":       f"{_REALSRC}/RealSource__Cook_rice_using_an_electric_rice_cooker",
    "hang_out_the_clothes_to":           f"{_REALSRC}/RealSource__Hang_out_the_clothes_to_dry",
    "lt_cleaning_and_organizing_of":     f"{_REALSRC}/RealSource__LT__Cleaning_and_organizing_of_books",
    "lt_conveyor_belt_sorts_and":        f"{_REALSRC}/RealSource__LT__Conveyor_belt_sorts_and_separates_different_parts",
    "lt_dishwasher_cleans_tableware":    f"{_REALSRC}/RealSource__LT__Dishwasher_cleans_tableware",
    "lt_fruit_shelving_and_arrangement": f"{_REALSRC}/RealSource__LT__Fruit_shelving_and_arrangement",
    "lt_storage_and_arrangement_bed":    f"{_REALSRC}/RealSource__LT__Storage_and_arrangement_of_clothes_on_the_bed",
    "lt_storage_and_arrangement_sofa":   f"{_REALSRC}/RealSource__LT__Storage_and_arrangement_of_clothes_on_the_sofa",
    "make_toast":                        f"{_REALSRC}/RealSource__Make_toast",
    "making_steamed_potatoes":           f"{_REALSRC}/RealSource__Making_steamed_potatoes",
    "move_industrial_parts":             f"{_REALSRC}/RealSource__Move_industrial_parts_to_different_plastic_boxes",
    "organize_the_glass_tube_on":        f"{_REALSRC}/RealSource__Organize_the_glass_tube_on_the_rack",
    "organize_the_magazines":            f"{_REALSRC}/RealSource__Organize_the_magazines",
    "organize_the_pen_holder":           f"{_REALSRC}/RealSource__Organize_the_pen_holder",
    "organize_the_repair_tools":         f"{_REALSRC}/RealSource__Organize_the_repair_tools",
    "organize_the_tv_cabinet":           f"{_REALSRC}/RealSource__Organize_the_TV_cabinet",
    "pack_the_badminton_shuttlecock":    f"{_REALSRC}/RealSource__Pack_the_badminton_shuttlecock",
    "place_the_books":                   f"{_REALSRC}/RealSource__Place_the_books",
    "place_the_hairdryer":               f"{_REALSRC}/RealSource__Place_the_hairdryer",
    "place_the_slippers":                f"{_REALSRC}/RealSource__Place_the_slippers",
    "prepare_the_birthday_cake":         f"{_REALSRC}/RealSource__Prepare_the_birthday_cake",
    "prepare_the_bread":                 f"{_REALSRC}/RealSource__Prepare_the_bread",
    "put_the_milk_in_the":               f"{_REALSRC}/RealSource__Put_the_milk_in_the_refrigerator",
    "refill_the_laundry_detergent":      f"{_REALSRC}/RealSource__Refill_the_laundry_detergent",
    "replace_the_tissues_and_arrange":   f"{_REALSRC}/RealSource__Replace_the_tissues_and_arrange_them",
    "replenish_tea_bags":                f"{_REALSRC}/RealSource__Replenish_tea_bags",
    "stack_the_cups":                    f"{_REALSRC}/RealSource__Stack_the_cups",
    "steam_buns":                        f"{_REALSRC}/RealSource__Steam_buns",
    "steaming_rice_in_a_rice":           f"{_REALSRC}/RealSource__Steaming_rice_in_a_rice_cooker",
    "take_down_the_book":                f"{_REALSRC}/RealSource__Take_down_the_book",
    "take_out_the_trash":                f"{_REALSRC}/RealSource__Take_out_the_trash",
    "tidy_up_the_childrens_room":        f"{_REALSRC}/RealSource__Tidy_up_the_childrens_room",
    "tidy_up_the_conference_room":       f"{_REALSRC}/RealSource__Tidy_up_the_conference_room_table",
    "tidy_up_the_cooking_counter":       f"{_REALSRC}/RealSource__Tidy_up_the_cooking_counter",
    "tidy_up_the_kitchen_counter":       f"{_REALSRC}/RealSource__Tidy_up_the_kitchen_counter",
}

_AGILEX = f"{_VL}/agilex_jd"
# Diverse sample across 18 scene families for task screening.
AGILEX_JD_TASKS = {
    "f01_classify_seasonings":  f"{_AGILEX}/agilex_cobotmagic2_dualArm-gripper-3cameras_1/1_classify_the_seasonings_take_out_the_seasonings_and_cook_the_dishes_20260326/success/lerobot_RoboMIND",
    "f03_h_d_03_02":            f"{_AGILEX}/agilex_cobotmagic2_dualArm-gripper-3cameras_3/agilex_cobotmagic2_dualArm-gripper-3cameras_3_H-D-03_02_01_20251027/success/lerobot_RoboMIND",
    "f06_count_banknotes":      f"{_AGILEX}/agilex_cobotmagic2_dualArm-gripper-3cameras_6/6_use_a_bill_counter_to_sort_and_flatten_banknotes/success/lerobot_RoboMIND",
    "f07_load_thermal_paper":   f"{_AGILEX}/agilex_cobotmagic2_dualArm-gripper-3cameras_7/SL7_load_the_new_thermal_paper_roll_into_the_paper_tray_in_the_correct_direction_and_leave_a_2_cm_paper_tab_0105/success/lerobot_RoboMIND",
    "f08_h_d_03_21":            f"{_AGILEX}/agilex_cobotmagic2_dualArm-gripper-3cameras_8/agilex_cobotmagic2_dualArm-gripper-3cameras_8_H-D-03_21_04_20251103/success/lerobot_RoboMIND",
    "f12_adjust_monitor_angle": f"{_AGILEX}/agilex_cobotmagic2_dualArm-gripper-3cameras_12/agilex_cobotmagic2_dualArm-gripper-3cameras_12_adjust_the_monitor_angle_0124/success/lerobot_RoboMIND",
    "f15_arrange_books":        f"{_AGILEX}/agilex_cobotmagic2_dualArm-gripper-3cameras_15/15_arrange_the_books_on_the_bookshelf_20251204/success/lerobot_RoboMIND",
    "f16_refill_rubber":        f"{_AGILEX}/agilex_cobotmagic2_dualArm-gripper-3cameras_16/16_refill_the_rubber_in_the_drawer_20251224/success/lerobot_RoboMIND",
    "f18_clean_filter_residue": f"{_AGILEX}/agilex_cobotmagic2_dualArm-gripper-3cameras_18/agilex_cobotmagic2_dualArm-gripper-3cameras_18_clean_the_residue_from_the_filter_1229/success/lerobot_RoboMIND",
    "f21_ep_20260115_pm":       f"{_AGILEX}/agilex_cobotmagic2_dualArm-gripper-3cameras_21/agilex_cobotmagic2_dualArm-gripper-3cameras_21_agilex_cobotmagic2_dualArm-gripper-3cameras_21_20260115_pm/success/lerobot_RoboMIND",
    "f25_arrange_cheese_plate": f"{_AGILEX}/agilex_cobotmagic2_dualArm-gripper-3cameras_25/agilex_cobotmagic2_dualArm-gripper-3cameras_25_arrange_a_plate_of_cheese_20260128_pm/success/lerobot_RoboMIND",
    "f30_arrange_bills":        f"{_AGILEX}/agilex_cobotmagic2_dualArm-gripper-3cameras_30/agilex_30_arrange_the_bills_in_order_of_denomination_and_place_them_into_the_compartments_of_the_cash_box_20260205_am/success/lerobot_RoboMIND",
    "f32_wash_kitchen_tableware": f"{_AGILEX}/agilex_cobotmagic2_dualArm-gripper-3cameras_32/agilex_cobotmagic2_dualArm-gripper-3cameras_32_home_kitchen_tableware_washing_and_drying_20260213_am/success/lerobot_RoboMIND",
    "f40_charge_devices":       f"{_AGILEX}/agilex_cobotmagic2_dualArm-gripper-3cameras_40/agilex_cobotmagic2_dualArm-gripper-3cameras_40_charge_electronic_devices_20260202_pm/success/lerobot_RoboMIND",
    "f43_flatten_coin_roll":    f"{_AGILEX}/agilex_cobotmagic2_dualArm-gripper-3cameras_43/agilex_cobotmagic2_dualArm-gripper-3cameras_43_flatten_the_empty_coin_roll_paper_and_recycle_it_20260116_pm/success/lerobot_RoboMIND",
    "f49_clean_toothbrush":     f"{_AGILEX}/agilex_cobotmagic2_dualArm-gripper-3cameras_49/agilex_cobotmagic2_dualArm-gripper-3cameras_49_cleaning_an_electric_toothbrush_20260314_am/success/lerobot_RoboMIND",
    "f50_change_light_bulb":    f"{_AGILEX}/agilex_cobotmagic2_dualArm-gripper-3cameras_50/agilex_cobotmagic2_dualArm-gripper-3cameras_50_change_a_light_bulb_20260321_pm/success/lerobot_RoboMIND",
    "f51_clean_office_desk":    f"{_AGILEX}/agilex_cobotmagic2_dualArm-gripper-3cameras_51/agilex_cobotmagic2_dualArm-gripper-3cameras_51_clean_the_office_desk_20260123_pm/success/lerobot_RoboMIND",
}

_FR3 = f"{_V3}/franka/franka_fr3_dualArm-gripper-6cameras_1"
FRANKA_FR3_TASKS = {
    "pick_lemon":  f"{_FR3}/pick_up_lemon_place_on_plate_250418/success/lerobot_RoboMIND",
    "pick_cookie": f"{_FR3}/pick_up_cookie_place_on_plate_250514/success/lerobot_RoboMIND",
    "pick_coke":   f"{_FR3}/pick_up_coke_bottle_place_on_plate_250515/success/lerobot_RoboMIND",
}

_FR_SINGLE = f"{_V3}/franka/franka_emika_singleArm-gripper-4cameras_2"
FRANKA_SINGLE_TASKS = {
    "cup_on_holder": f"{_FR_SINGLE}/cup_goes_on_the_cup_holder/success/lerobot_RoboMIND",
}


# ── Embodiment registry ──────────────────────────────────────────────────────
EMBODIMENTS: dict[str, Embodiment] = {
    # LIBERO — single arm 8D, 2 cameras (hdf5)
    "libero": Embodiment(
        name="libero", arm="single", state_dim=8, loader="hdf5_libero",
        cameras=[Camera("agentview", role="third_person"),
                 Camera("wrist", role="wrist")],
    ),
    # RoboTwin — dual arm 14D, 3 cameras (hdf5)
    "robotwin": Embodiment(
        name="robotwin", arm="dual", state_dim=14, loader="hdf5_robotwin",
        cameras=[Camera("cam_high", role="third_person"),
                 Camera("cam_left_wrist", role="wrist"),
                 Camera("cam_right_wrist", role="wrist")],
    ),
    # Tienkung — dual arm 16D, 3 cameras (top/left/right)
    "tienkung": Embodiment(
        name="tienkung", arm="dual", state_dim=16, loader="lerobot",
        cameras=[Camera("top",   "camera_observations.color_images.camera_top",   "third_person"),
                 Camera("left",  "camera_observations.color_images.camera_left",  "aux"),
                 Camera("right", "camera_observations.color_images.camera_right", "aux")],
        state_cols=[("puppet.arm_left_position_align.data", 7),
                    ("puppet.end_effector_left_position_align.data", 1),
                    ("puppet.arm_right_position_align.data", 7),
                    ("puppet.end_effector_right_position_align.data", 1)],
        tasks=TIENKUNG_TASKS, default_suite="organize_tools",
    ),
    # UR5e — single arm 7D, 3 cameras (front/right/waist)
    "ur5e": Embodiment(
        name="ur5e", arm="single", state_dim=7, loader="lerobot",
        cameras=[Camera("front", "camera_observations.color_images.camera_front", "third_person"),
                 Camera("right", "camera_observations.color_images.camera_right", "aux"),
                 Camera("waist", "camera_observations.color_images.camera_waist", "aux")],
        state_cols=[("puppet.arm_single_position_align.data", 6),
                    ("puppet.end_effector_single_position_align.data", 1)],
        tasks=UR5E_TASKS, default_suite="install_gears",
    ),
    # UR5e v3 — single arm 7D, 3 third-person cameras (front/left/right)
    "ur5e_v3": Embodiment(
        name="ur5e_v3", arm="single", state_dim=7, loader="lerobot",
        cameras=[Camera("front", "camera_observations.color_images.camera_front", "third_person"),
                 Camera("left",  "camera_observations.color_images.camera_left",  "aux"),
                 Camera("right", "camera_observations.color_images.camera_right", "aux")],
        state_cols=[("puppet.arm_single_position_align.data", 6),
                    ("puppet.end_effector_single_position_align.data", 1)],
        tasks=UR5E_V3_TASKS, default_suite="pick_bread",
    ),
    # UR5e dual — dual arm 14D, top/front/right
    "ur5e_dual": Embodiment(
        name="ur5e_dual", arm="dual", state_dim=14, loader="lerobot",
        cameras=[Camera("top",   "camera_observations.color_images.camera_top",   "third_person"),
                 Camera("front", "camera_observations.color_images.camera_front", "third_person"),
                 Camera("right", "camera_observations.color_images.camera_right", "aux")],
        state_cols=[("puppet.arm_left_position_align.data", 6),
                    ("puppet.end_effector_left_position_align.data", 1),
                    ("puppet.arm_right_position_align.data", 6),
                    ("puppet.end_effector_right_position_align.data", 1)],
        tasks=UR5E_DUAL_TASKS, default_suite="add_ice",
    ),
    # Agilex JD (松灵) — dual arm 14D, head + wrist_left + wrist_right
    "agilex_jd": Embodiment(
        name="agilex_jd", arm="dual", state_dim=14, loader="lerobot",
        cameras=[Camera("head_camera",       "camera_observations.color_images.camera_head",        "third_person"),
                 Camera("left_hand_camera",  "camera_observations.color_images.camera_wrist_left",  "wrist"),
                 Camera("right_hand_camera", "camera_observations.color_images.camera_wrist_right", "wrist")],
        state_cols=[("puppet.arm_left_position_align.data", 6),
                    ("puppet.end_effector_left_position_align.data", 1),
                    ("puppet.arm_right_position_align.data", 6),
                    ("puppet.end_effector_right_position_align.data", 1)],
        tasks=AGILEX_JD_TASKS, default_suite="f01_classify_seasonings",
    ),
    # RealSource World v3.0 — dual arm 16D, head + left_hand + right_hand (LeRobot v2)
    "realsource_v30": Embodiment(
        name="realsource_v30", arm="dual", state_dim=16, loader="lerobot",
        cameras=[Camera("head_camera",       "observation.images.head_camera",       "third_person"),
                 Camera("left_hand_camera",  "observation.images.left_hand_camera",  "wrist"),
                 Camera("right_hand_camera", "observation.images.right_hand_camera", "wrist")],
        state_cols=[("observation.state", 16)],
        instruction_source="episode_tasks",
        tasks=REALSOURCE_V30_TASKS, default_suite="arrange_the_cups",
    ),
    # Franka fr3 — dual arm 18D (arm[8]+grip[1] per side), 3 cameras.
    # NOTE: on the fr3 dualArm-gripper-6cameras_1 datasets we have, the
    # `camera_right` chunk is empty; use top/front/left instead.
    "franka_fr3": Embodiment(
        name="franka_fr3", arm="dual", state_dim=18, loader="lerobot",
        cameras=[Camera("top",   "camera_observations.color_images.camera_top",   "third_person"),
                 Camera("front", "camera_observations.color_images.camera_front", "third_person"),
                 Camera("left",  "camera_observations.color_images.camera_left",  "aux")],
        state_cols=[("puppet.arm_left_position_align.data", 8),
                    ("puppet.end_effector_left_position_align.data", 1),   # gripper (may be missing)
                    ("puppet.arm_right_position_align.data", 8),
                    ("puppet.end_effector_right_position_align.data", 1)],
        tasks=FRANKA_FR3_TASKS, default_suite="pick_lemon",
    ),
    # Franka Emika — single arm. Nominally 4 cameras (top/wrist/left/right) but
    # on the cup_on_holder dataset the `camera_wrist` stream is empty, so we use
    # the three that ship: top/left/right. These datasets also carry videos +
    # episode metadata but no state parquet, so state reads back as zeros (the
    # adapter zero-pads to its expected dim). All three cameras feed the model.
    "franka_single": Embodiment(
        name="franka_single", arm="single", state_dim=5, loader="lerobot",
        cameras=[Camera("top",   "camera_observations.color_images.camera_top",   "third_person"),
                 Camera("left",  "camera_observations.color_images.camera_left",  "aux"),
                 Camera("right", "camera_observations.color_images.camera_right", "aux")],
        state_cols=[("puppet.arm_left_position_align.data", 4),
                    ("puppet.end_effector_single_position_align.data", 1)],
        instruction_source="episode_tasks",
        tasks=FRANKA_SINGLE_TASKS, default_suite="cup_on_holder",
    ),
}


# PelicanVLA 05 release stats.json packs AgileX Split Aloha state into 14D
# (left arm + left grip + right arm + right grip) — this is the model's
# training-side convention, single source of truth here.
PELICANVLA_AGILEX_STATE_FIELDS = [
    "states.left_joint.position", "states.left_gripper.position",
    "states.right_joint.position", "states.right_gripper.position",
]


def get(name: str) -> Embodiment:
    if name not in EMBODIMENTS:
        raise KeyError(
            f"unknown embodiment/source {name!r}; registered: {list(EMBODIMENTS)}. "
            "To add one, append an Embodiment to EMBODIMENTS in this file.")
    return EMBODIMENTS[name]


def lerobot_datasets() -> dict:
    """Derived config for `sources/_lerobot_frame.py` (loader == 'lerobot').

    `root` defaults to the `default_suite` entry; the RoboMIND source overrides
    it per suite at read time via `_set_root()`.
    """
    out = {}
    for name, e in EMBODIMENTS.items():
        if e.loader != "lerobot":
            continue
        default_root = e.tasks.get(e.default_suite, "") if e.default_suite else ""
        cfg = {
            "root": default_root,
            "cameras": e.cams,
            "camera_keys": e.camera_keys,
            "state_cols": list(e.state_cols or []),
        }
        if e.instruction_source:
            cfg["instruction_source"] = e.instruction_source
        out[name] = cfg
    return out


def camera_role_sets() -> tuple[set[str], set[str]]:
    """(third_person short names, wrist short names) — used by preflight.

    "aux" doesn't join either set.
    """
    third, wrist = set(), set()
    for e in EMBODIMENTS.values():
        for c in e.cameras:
            if c.role == "third_person":
                third.add(c.short)
            elif c.role == "wrist":
                wrist.add(c.short)
    return third, wrist


def check_consistency() -> list[str]:
    problems = []
    third, wrist = camera_role_sets()
    dup = third & wrist
    if dup:
        problems.append(f"camera short name is both third_person and wrist: {sorted(dup)}")
    for name, e in EMBODIMENTS.items():
        if e.loader == "lerobot":
            got = sum(d for _, d in (e.state_cols or []))
            if got != e.state_dim:
                problems.append(f"{name}: state_cols sum {got} != state_dim {e.state_dim}")
            if e.default_suite and e.default_suite not in e.tasks:
                problems.append(f"{name}: default_suite {e.default_suite!r} not in tasks")
    return problems
