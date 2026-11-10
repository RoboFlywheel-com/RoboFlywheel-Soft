# SPDX-License-Identifier: Apache-2.0
"""Representation selection for manual mesh-adapter review, independent of gates.

Object semantics precede the coarse dataset family. All choices and surrogate
assumptions are explicit; this is not an identified material parameter model.
"""
from __future__ import annotations


def select_representation(asset):
    family = asset.get('geometry_family') or asset.get('route', {}).get('geometry_family', 'solid')
    name = str(asset.get('object_type') or '').lower()
    material = str(asset.get('material_family') or '').lower()
    def has(*words):
        return any(w in name for w in words)

    profile, reason = 'filled_volume', 'solid-like object with interior material'
    if has('inflatable', 'air pillow', 'blood bag', 'urine bag', 'refill pouch'):
        profile, reason = 'sealed_membrane', 'sealed hollow envelope or fluid-filled container'
    elif family == 'cloth_sheet' or has('scarf', 'webbing', 'fabric seat', 'plastic film', 'shoulder strap', 'hip belt'):
        profile, reason = 'sheet', 'intrinsically sheet-like object'
    elif family == 'garment':
        profile = ('reinforced_shell' if has('padded', 'down jacket', 'cap', 'hat', 'bra') else
                   'garment_cloth' if material == 'textile' else 'garment_shell')
        reason = ('assembled textile surface with negligible bending support' if profile == 'garment_cloth'
                  else 'preserve assembled garment geometry and local rest curvature')
    elif has('backpack', 'shopping bag', 'organizer pouch', 'paper bag', 'blister pack', 'envelope'):
        profile, reason = 'reinforced_shell', 'preformed container walls with bending support; empty cavity'
    elif has('wrapper', 'mailer', 'document pouch', 'anti-static bag', 'chip bag'):
        profile, reason = 'bag_shell', 'collapsible container walls; retain separated front and back surfaces'
    elif has('cover', 'finger puppet', 'silicone liner', 'silicone membrane', 'blood vessel', 'medical tubing'):
        profile, reason = 'garment_shell', 'thin hollow object; do not fill its cavity with MPM particles'
    elif family == 'bag' and not has('bean', 'sandbag', 'sack', 'flour', 'foam', 'mud', 'snow', 'tissue', 'food pack', 'sleeping bag'):
        profile, reason = 'bag_shell', 'container family with no evidence of a bulk filling'
    elif family == 'shell' and not has('cushion', 'pillow', 'pad', 'foam', 'pet bed', 'futon', 'beanbag'):
        profile, reason = 'reinforced_shell', 'preformed hollow shell, not a flat membrane'
    # Unknown/unsupported historical routes are still visualized as an explicit
    # approximation. Neither damage nor fluid microstructure is claimed here.
    override = asset.get('adaptation_review', {}).get('representation')
    if override:
        if override not in {'filled_volume', 'sheet', 'garment_cloth', 'garment_shell', 'bag_shell', 'reinforced_shell', 'sealed_membrane'}:
            raise ValueError(f'unknown representation override: {override}')
        profile, reason = override, 'explicit asset representation override'
    bending = {'sheet': None, 'garment_cloth': None, 'garment_shell': 2e-3, 'bag_shell': 4e-4,
               'reinforced_shell': 1e-5, 'sealed_membrane': 5e-4}.get(profile)
    return {'version': 'v5-review', 'profile': profile, 'reason': reason,
            'geometry_family': family, 'object_type': asset.get('object_type'), 'material_family': material,
            'solver': 'MPM' if profile == 'filled_volume' else 'PBD',
            'cavity': 'none_or_homogenized_filling' if profile == 'filled_volume' else
                      'closed_volume_constraint' if profile == 'sealed_membrane' else
                      'none' if profile == 'sheet' else 'empty_hollow_shell',
            'rest_curvature': profile not in {'filled_volume', 'sheet', 'garment_cloth'},
            'bending_compliance': bending,
            'stretch_compliance': 1e-7,
            'scope': 'representation and solver approximation for human review; not material identification'}


def refine_for_geometry(representation, vertices):
    """Do not flatten a full 3D object solely because its catalog name says strap.

    This guard applies to sheet suggestions outside the actual cloth_sheet
    family. It preserves ambiguity as a 3D shell; it does not identify an object
    or reconstruct its missing pattern. Explicit user overrides take priority.
    """
    if (representation['profile']!='sheet' or representation['geometry_family']=='cloth_sheet'
            or representation['reason']=='explicit asset representation override'):
        return representation
    import numpy as np
    points=np.asarray(vertices,dtype=float)
    _,axes=np.linalg.eigh(np.cov(points.T))
    spans=np.ptp(points@axes,axis=0)
    ratio=float(spans.min()/max(spans.max(),1e-12))
    if ratio<=.18:return representation
    profile=('reinforced_shell' if 'backpack' in str(representation.get('object_type','')).lower() else
             'garment_cloth' if representation['geometry_family']=='garment' and
             representation['material_family']=='textile' else 'garment_shell')
    refined=dict(representation,profile=profile,rest_curvature=profile!='garment_cloth',cavity='empty_hollow_shell',
                 bending_compliance=1e-5 if profile=='reinforced_shell' else None if profile=='garment_cloth' else 2e-3,
                 stretch_compliance=1e-7,
                 reason='nonplanar generated geometry conflicts with a sheet-like catalog name; preserve 3D boundary',
                 semantic_geometry_conflict={'suggested_profile':'sheet','principal_extent_ratio':ratio,
                     'sheet_ratio_max':.18,'policy':'preserve generated 3D shape for human review'})
    return refined
